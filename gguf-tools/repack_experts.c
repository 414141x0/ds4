/*
 * repack_experts.c - Extract expert weights from a DeepSeek V4 GGUF into
 * per-layer packed binary files for SSD-to-GPU streaming.
 *
 * Output: packed_experts/layer_XX.bin  (XX = 00..42)
 *         packed_experts/manifest.json
 *
 * Each layer file contains 256 experts laid out contiguously:
 *   expert 0: [gate_data][up_data][down_data][padding_to_4K]
 *   expert 1: [gate_data][up_data][down_data][padding_to_4K]
 *   ...
 *
 * Usage: repack_experts <model.gguf> [output_dir]
 */

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdint.h>
#include <stdbool.h>
#include <inttypes.h>

#define DS4_N_LAYER  43
#define DS4_N_EXPERT 256
#define QK_K         256
#define DS4_MAX_DIMS 4
#define ALIGN_4K     4096

enum {
    GGUF_VALUE_UINT8   = 0,
    GGUF_VALUE_INT8    = 1,
    GGUF_VALUE_UINT16  = 2,
    GGUF_VALUE_INT16   = 3,
    GGUF_VALUE_UINT32  = 4,
    GGUF_VALUE_INT32   = 5,
    GGUF_VALUE_FLOAT32 = 6,
    GGUF_VALUE_BOOL    = 7,
    GGUF_VALUE_STRING  = 8,
    GGUF_VALUE_ARRAY   = 9,
    GGUF_VALUE_UINT64  = 10,
    GGUF_VALUE_INT64   = 11,
    GGUF_VALUE_FLOAT64 = 12,
};

enum {
    TENSOR_F16      = 1,
    TENSOR_Q8_0     = 8,
    TENSOR_Q2_K     = 10,
    TENSOR_Q4_K     = 12,
    TENSOR_IQ2_XXS  = 16,
};

typedef struct { const char *ptr; uint64_t len; } str_ref;

typedef struct {
    str_ref name;
    uint32_t ndim;
    uint64_t dim[DS4_MAX_DIMS];
    uint32_t type;
    uint64_t abs_offset;
    uint64_t bytes;
} tensor_info;

typedef struct {
    const uint8_t *map;
    uint64_t size;
    uint64_t pos;
} cursor;

static void die(const char *msg) {
    fprintf(stderr, "repack_experts: %s\n", msg);
    exit(1);
}

static bool cur_read(cursor *c, void *dst, uint64_t n) {
    if (c->pos + n > c->size) return false;
    memcpy(dst, c->map + c->pos, (size_t)n);
    c->pos += n;
    return true;
}

static bool cur_u32(cursor *c, uint32_t *v) { return cur_read(c, v, 4); }
static bool cur_u64(cursor *c, uint64_t *v) { return cur_read(c, v, 8); }

static bool cur_string(cursor *c, str_ref *s) {
    if (!cur_u64(c, &s->len)) return false;
    if (c->pos + s->len > c->size) return false;
    s->ptr = (const char *)(c->map + c->pos);
    c->pos += s->len;
    return true;
}

static uint64_t scalar_size(uint32_t type) {
    switch (type) {
    case GGUF_VALUE_UINT8: case GGUF_VALUE_INT8: case GGUF_VALUE_BOOL: return 1;
    case GGUF_VALUE_UINT16: case GGUF_VALUE_INT16: return 2;
    case GGUF_VALUE_UINT32: case GGUF_VALUE_INT32: case GGUF_VALUE_FLOAT32: return 4;
    case GGUF_VALUE_UINT64: case GGUF_VALUE_INT64: case GGUF_VALUE_FLOAT64: return 8;
    default: return 0;
    }
}

static bool skip_value(cursor *c, uint32_t type, int depth);

static bool skip_value(cursor *c, uint32_t type, int depth) {
    if (depth > 8) return false;
    if (type == GGUF_VALUE_STRING) {
        str_ref s;
        return cur_string(c, &s);
    }
    if (type == GGUF_VALUE_ARRAY) {
        uint32_t elem_type;
        uint64_t len;
        if (!cur_u32(c, &elem_type) || !cur_u64(c, &len)) return false;
        for (uint64_t i = 0; i < len; i++) {
            if (!skip_value(c, elem_type, depth + 1)) return false;
        }
        return true;
    }
    uint64_t s = scalar_size(type);
    if (s == 0) return false;
    c->pos += s;
    return c->pos <= c->size;
}

static uint64_t align_up(uint64_t v, uint64_t a) {
    return (v + a - 1) / a * a;
}

static uint64_t block_bytes(uint32_t type) {
    switch (type) {
    case TENSOR_IQ2_XXS: return 2 + 2 * (QK_K / 8);  /* 66 */
    case TENSOR_Q2_K:    return 16 + 64 + 2 + 2;       /* 84 */
    case TENSOR_Q4_K:    return 2 + 2 + 12 + QK_K / 2; /* 144 */
    default: return 0;
    }
}

static uint64_t expert_row_bytes(uint32_t type, uint64_t in_dim) {
    uint64_t bb = block_bytes(type);
    if (bb == 0) die("unsupported quant type for expert tensor");
    return (in_dim / QK_K) * bb;
}

static uint64_t expert_total_bytes(uint32_t type, uint64_t in_dim, uint64_t out_dim) {
    return out_dim * expert_row_bytes(type, in_dim);
}

static bool str_match(str_ref s, const char *lit) {
    size_t ll = strlen(lit);
    return s.len == ll && memcmp(s.ptr, lit, ll) == 0;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: repack_experts <model.gguf> [output_dir]\n");
        return 1;
    }
    const char *gguf_path = argv[1];
    const char *out_dir = argc >= 3 ? argv[2] : "packed_experts";

    int fd = open(gguf_path, O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }
    struct stat st;
    if (fstat(fd, &st) < 0) { perror("fstat"); close(fd); return 1; }
    uint64_t file_size = (uint64_t)st.st_size;
    const uint8_t *map = mmap(NULL, (size_t)file_size, PROT_READ, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) { perror("mmap"); close(fd); return 1; }

    cursor c = { .map = map, .size = file_size, .pos = 0 };

    /* Parse GGUF header */
    uint32_t magic;
    if (!cur_u32(&c, &magic) || magic != 0x46554747) die("not a GGUF file");
    uint32_t version;
    if (!cur_u32(&c, &version)) die("cannot read version");
    if (version < 2 || version > 3) die("unsupported GGUF version");
    uint64_t n_tensors = 0, n_kv = 0;
    if (!cur_u64(&c, &n_tensors) || !cur_u64(&c, &n_kv)) die("cannot read header counts");

    fprintf(stderr, "GGUF: version=%u, tensors=%" PRIu64 ", kv=%" PRIu64 "\n", version, n_tensors, n_kv);

    /* Skip metadata */
    uint32_t alignment = 32;
    for (uint64_t i = 0; i < n_kv; i++) {
        str_ref key;
        uint32_t type = 0;
        if (!cur_string(&c, &key) || !cur_u32(&c, &type)) die("metadata parse error");
        if (str_match(key, "general.alignment") && type == GGUF_VALUE_UINT32) {
            cursor tmp = c;
            uint32_t a;
            if (cur_u32(&tmp, &a) && a != 0) alignment = a;
        }
        if (!skip_value(&c, type, 0)) die("metadata skip error");
    }

    /* Parse tensor directory */
    tensor_info *tensors = calloc((size_t)n_tensors, sizeof(tensor_info));
    if (!tensors) die("oom");
    for (uint64_t i = 0; i < n_tensors; i++) {
        tensor_info *t = &tensors[i];
        if (!cur_string(&c, &t->name)) die("tensor name parse error");
        if (!cur_u32(&c, &t->ndim)) die("tensor ndim parse error");
        if (t->ndim == 0 || t->ndim > DS4_MAX_DIMS) die("bad ndim");
        for (uint32_t d = 0; d < t->ndim; d++) {
            if (!cur_u64(&c, &t->dim[d])) die("tensor dim parse error");
        }
        if (!cur_u32(&c, &t->type)) die("tensor type parse error");
        uint64_t rel_offset;
        if (!cur_u64(&c, &rel_offset)) die("tensor offset parse error");

        /* We'll fix offsets after parsing all tensors since we need final c.pos */
        t->abs_offset = rel_offset; /* store relative for now */
    }
    uint64_t tensor_data_pos = align_up(c.pos, alignment);
    for (uint64_t i = 0; i < n_tensors; i++) {
        tensors[i].abs_offset += tensor_data_pos;
    }

    fprintf(stderr, "Tensor data starts at offset %" PRIu64 " (%.2f GiB)\n",
            tensor_data_pos, (double)tensor_data_pos / (1024.0*1024.0*1024.0));

    /* Create output directory */
    mkdir(out_dir, 0755);

    /* Find and repack expert tensors for each layer */
    char name_buf[256];
    char path_buf[512];

    /* We'll collect manifest info */
    uint32_t gate_type = 0, down_type = 0;
    uint64_t gate_eb = 0, up_eb = 0, down_eb = 0;
    uint64_t expert_stride = 0;

    for (int il = 0; il < DS4_N_LAYER; il++) {
        /* Find the 3 expert tensors for this layer */
        tensor_info *gate_t = NULL, *up_t = NULL, *down_t = NULL;

        snprintf(name_buf, sizeof(name_buf), "blk.%d.ffn_gate_exps.weight", il);
        for (uint64_t i = 0; i < n_tensors; i++) {
            if (str_match(tensors[i].name, name_buf)) { gate_t = &tensors[i]; break; }
        }
        snprintf(name_buf, sizeof(name_buf), "blk.%d.ffn_up_exps.weight", il);
        for (uint64_t i = 0; i < n_tensors; i++) {
            if (str_match(tensors[i].name, name_buf)) { up_t = &tensors[i]; break; }
        }
        snprintf(name_buf, sizeof(name_buf), "blk.%d.ffn_down_exps.weight", il);
        for (uint64_t i = 0; i < n_tensors; i++) {
            if (str_match(tensors[i].name, name_buf)) { down_t = &tensors[i]; break; }
        }

        if (!gate_t || !up_t || !down_t) {
            fprintf(stderr, "Layer %d: missing expert tensors, skipping\n", il);
            continue;
        }

        if (gate_t->ndim != 3 || up_t->ndim != 3 || down_t->ndim != 3)
            die("expert tensor is not 3D");
        if (gate_t->dim[2] != DS4_N_EXPERT || up_t->dim[2] != DS4_N_EXPERT || down_t->dim[2] != DS4_N_EXPERT)
            die("expert tensor does not have 256 experts");

        uint64_t g_eb = expert_total_bytes(gate_t->type, gate_t->dim[0], gate_t->dim[1]);
        uint64_t u_eb = expert_total_bytes(up_t->type,   up_t->dim[0],  up_t->dim[1]);
        uint64_t d_eb = expert_total_bytes(down_t->type, down_t->dim[0], down_t->dim[1]);
        uint64_t raw_stride = g_eb + u_eb + d_eb;
        uint64_t stride = align_up(raw_stride, ALIGN_4K);

        if (il == 0) {
            gate_type = gate_t->type;
            down_type = down_t->type;
            gate_eb = g_eb;
            up_eb = u_eb;
            down_eb = d_eb;
            expert_stride = stride;
            fprintf(stderr, "Expert sizes: gate=%" PRIu64 " up=%" PRIu64 " down=%" PRIu64
                    " raw=%" PRIu64 " stride=%" PRIu64 " (%.2f MB)\n",
                    g_eb, u_eb, d_eb, raw_stride, stride, (double)stride / (1024.0*1024.0));
        }

        snprintf(path_buf, sizeof(path_buf), "%s/layer_%02d.bin", out_dir, il);
        int ofd = open(path_buf, O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (ofd < 0) { perror(path_buf); return 1; }

#ifdef __APPLE__
        /* Request contiguous allocation on APFS */
        struct fstore fst = {
            .fst_flags = F_ALLOCATECONTIG | F_ALLOCATEALL,
            .fst_posmode = F_PEOFPOSMODE,
            .fst_offset = 0,
            .fst_length = (off_t)(stride * DS4_N_EXPERT),
        };
        fcntl(ofd, F_PREALLOCATE, &fst);
#endif
        if (ftruncate(ofd, (off_t)(stride * DS4_N_EXPERT)) < 0) {
            perror("ftruncate"); close(ofd); return 1;
        }

        const uint8_t *gate_base = map + gate_t->abs_offset;
        const uint8_t *up_base   = map + up_t->abs_offset;
        const uint8_t *down_base = map + down_t->abs_offset;

        /* Padding buffer for alignment */
        uint64_t pad_size = stride - raw_stride;
        uint8_t *pad = NULL;
        if (pad_size > 0) {
            pad = calloc(1, (size_t)pad_size);
            if (!pad) die("oom for padding");
        }

        for (int e = 0; e < DS4_N_EXPERT; e++) {
            const uint8_t *gd = gate_base + (uint64_t)e * g_eb;
            const uint8_t *ud = up_base   + (uint64_t)e * u_eb;
            const uint8_t *dd = down_base + (uint64_t)e * d_eb;

            if (write(ofd, gd, (size_t)g_eb) != (ssize_t)g_eb) die("write gate failed");
            if (write(ofd, ud, (size_t)u_eb) != (ssize_t)u_eb) die("write up failed");
            if (write(ofd, dd, (size_t)d_eb) != (ssize_t)d_eb) die("write down failed");
            if (pad_size > 0) {
                if (write(ofd, pad, (size_t)pad_size) != (ssize_t)pad_size) die("write pad failed");
            }
        }

        close(ofd);
        free(pad);
        fprintf(stderr, "Layer %2d: wrote %s (%.2f GB)\n", il, path_buf,
                (double)(stride * DS4_N_EXPERT) / (1024.0*1024.0*1024.0));
    }

    /* Write manifest */
    snprintf(path_buf, sizeof(path_buf), "%s/manifest.json", out_dir);
    FILE *mf = fopen(path_buf, "w");
    if (!mf) { perror("manifest.json"); return 1; }
    fprintf(mf,
        "{\n"
        "  \"n_layer\": %d,\n"
        "  \"n_expert\": %d,\n"
        "  \"expert_stride\": %" PRIu64 ",\n"
        "  \"gate_bytes\": %" PRIu64 ",\n"
        "  \"up_bytes\": %" PRIu64 ",\n"
        "  \"down_bytes\": %" PRIu64 ",\n"
        "  \"gate_type\": %u,\n"
        "  \"down_type\": %u,\n"
        "  \"gate_in_dim\": 4096,\n"
        "  \"gate_out_dim\": 2048,\n"
        "  \"down_in_dim\": 2048,\n"
        "  \"down_out_dim\": 4096\n"
        "}\n",
        DS4_N_LAYER, DS4_N_EXPERT, expert_stride,
        gate_eb, up_eb, down_eb, gate_type, down_type);
    fclose(mf);
    fprintf(stderr, "Wrote %s\n", path_buf);

    munmap((void *)map, (size_t)file_size);
    close(fd);
    free(tensors);

    fprintf(stderr, "Done. Total packed size: %.2f GB\n",
            (double)(expert_stride * DS4_N_EXPERT * DS4_N_LAYER) / (1024.0*1024.0*1024.0));
    return 0;
}
