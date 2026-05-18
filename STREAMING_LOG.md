# Expert Streaming Implementation Log

## Assumptions
1. Expert weights in GGUF are laid out as 3D tensors: `ffn_gate_exps[in_dim][out_dim][256]`, same for up/down
2. The existing `mul_mv_id` kernels use `selected` buffer to index into expert tensors - for streaming we remap selected to [0..5] and pack 6 experts into combined staging buffers
3. 2MB alignment for DMA buffers based on Flash-iOS benchmarks (3.6x throughput improvement)
4. OS page cache (CLOCK-Pro on macOS) outperforms custom LRU caching per Flash-iOS findings
5. Prefill path will fall back to mmap (v1) - prefill warms page cache for decode
6. Single-instance engine means file-scope `g_stream_ctx` is safe (ds4 uses global lock)

## Implementation Progress
- [x] Phase 1: Expert repack tool (`gguf-tools/repack_experts.c`)
- [x] Phase 2: I/O thread pool (`ds4_expert_io.h`, `ds4_expert_io.c`)
- [x] Phase 3: DMA buffer allocation in Metal (`ds4_metal.m`: `ds4_gpu_init_expert_streaming`)
- [x] Phase 4: Streamed MoE kernel dispatch (`ds4_metal.m`: `ds4_gpu_routed_moe_one_streamed`)
- [x] Phase 5: Modified decode layer pipeline (`ds4.c`: streaming branch in `metal_graph_encode_decode_layer`)
- [x] Phase 6: Engine integration (`ds4.c`: `expert_streaming_init/close`, `ds4.h` options)
- [x] Phase 7: CLI integration + Makefile (`ds4_cli.c`: `--expert-pack-dir`, Makefile updated)
- [x] Phase 7b: Selective model mapping — only map ~8GB non-expert spans as Metal buffers (`compute_non_expert_spans`, `ds4_gpu_set_model_map_spans`)
- [x] Phase 7c: Prefill streaming branch (`metal_graph_encode_layer_ffn_batch` streaming path)
- [ ] Phase 8: End-to-end testing with real model
- [ ] Phase 9: Performance tuning

## Build Status
- All binaries compile and link cleanly: `ds4`, `ds4-server`, `ds4-bench`, `ds4-eval`, `repack-experts`
- Test binary links successfully (needs model file for runtime tests)
- Zero warnings in core code

## Errors & Corrections
1. **Forward declaration of engine struct** - `metal_graph_encode_decode_layer()` couldn't access `ds4_engine` members because it's defined later in the file. Fixed by extracting streaming state into `ds4_expert_stream_ctx` file-scope global.
2. **Static function ordering in ds4_metal.m** - `ds4_gpu_routed_moe_one_streamed()` was initially placed too early (after `ds4_gpu_wrap_model_range`), before helper functions it calls. Relocated after `ds4_gpu_routed_moe_one_tensor()` (line ~13162).
3. **Staging buffer approach** - Instead of dispatching 6 individual single-expert matmuls (which would lose fused pair_swiglu optimization), we memcpy from 6 DMA buffers into 3 combined staging Metal buffers (gate/up/down) and remap selected indices to [0..5]. This reuses all existing kernel optimizations (pair_swiglu fusion, sum6 down projection).
4. **Kernel panic root cause** - `ds4_engine_open()` called `expert_streaming_init()` AFTER Metal model mapping, so `e->expert_streaming` was always false at mapping time. Moved streaming init before mapping. Also: `newBufferWithBytesNoCopy` over 82GB on 48GB machine causes kernel watchdog timeout during VM page-table setup, regardless of residency settings. Fixed by selective mapping of ~8GB non-expert spans only.
5. **Prefill path crash** - `ds4_gpu_routed_moe_batch_tensor()` calls `ds4_gpu_wrap_model_range()` with expert offsets that aren't in mapped Metal views when streaming. Fixed by adding a per-token streaming branch in `metal_graph_encode_layer_ffn_batch()` that reuses the decode streaming path (pread + `ds4_gpu_routed_moe_one_streamed`).

## Architecture Notes
- DMA buffers: 6 slots × expert_stride bytes, 2MB-aligned, `newBufferWithBytesNoCopy` shared mode
- Staging buffers: 3 combined (gate_combined, up_combined, down_combined), allocated lazily on first streamed dispatch
- Pipeline per layer: GPU(attention+router) → end_commands → CPU readback selected[6] → 6×pread parallel → memcpy into staging → begin_commands → GPU(MoE from staging)
- The `mul_mv_id` kernels see `ne02=6` (6 experts in buffer) with selected=[0,1,2,3,4,5]
