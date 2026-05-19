# ds4 Optimization Log — M4 Pro 48GB

## Implemented Optimizations

### Session 1: I/O-GPU Overlap
| # | Optimization | Measured Impact | Status |
|---|-------------|----------------|--------|
| 1 | Within-layer I/O-GPU overlap (decode): split gate+up/down with flush_commands | +8.1% decode (3.2→3.46 t/s) | DONE |
| 2 | Within-layer I/O-GPU overlap (prefill): same pattern | Working, hard to measure in isolation | DONE |
| 3 | Branchless XOR sign trick in IQ2_XXS shader | Noise-neutral (compiler already optimized) | DONE |

### Session 2: Shader + I/O + Driver Optimizations
| # | Optimization | Measured Impact | Status |
|---|-------------|----------------|--------|
| 4 | MTLMathModeRelaxed on shader compilation | Part of +6.4% combined gain | DONE |
| 5 | Constant-memory LUT (eliminated threadgroup copy + barrier) | Part of +6.4% combined gain (3.46→3.68 t/s) | DONE |
| 6 | Strided F_RDADVISE at 512KB chunks (was only prefetching 20% of each expert) | Within bench noise, benefits longer generation | DONE |
| 7 | F_RDAHEAD on model fd | Enables OS-level sequential read-ahead | DONE |
| 8 | Command queue with low-latency shared events + disabled cross-queue hazard tracking | Within bench noise, reduces sync latency | DONE |
| 9 | Threadgroup memory freed (smem=0 for IQ2_XXS) | Allows higher GPU occupancy | DONE |

### Cumulative: ~3.2 → ~3.65 t/s decode (+14%), ~7.4 t/s prefill (128 tok)

## Rejected Optimizations (with reasons)

| Optimization | Why Rejected |
|-------------|-------------|
| F_NOCACHE on model fd | Page cache delivers 50-60% hit rate — disabling would nearly double I/O load |
| N_R0_IQ2_XXS=8 (8 rows/simdgroup) | 12% regression — reduced occupancy from higher register pressure |
| ASTC texture compression for weights | Lossy visual compression, incompatible with IQ2_XXS codebook |
| MPP/tensor_ops matmul2d | M5-only (Metal 4), not available on M4 Pro |
| CoreML quantized inference | Only supports 4/8-bit linear, not IQ2_XXS codebook format |
| MPS matrix-vector multiply | No sub-byte quantization support |

## Performance Blockers (Hard Limits)

1. **SSD bandwidth (~7.5 GB/s)** — Fundamental bottleneck for cold expert reads on 48GB system
2. **No hardware IQ2_XXS decompression** — Apple's hardware targets int4/int8 blockwise
3. **UMA = no SSD-to-GPU bypass** — CPU and GPU share physical memory
4. **Decode is memory-bandwidth-bound (M=1)** — Reducing bytes or increasing BW are the only levers
5. **kern.speculative_prefetch_max_iosize = 512KB** — Limits F_RDADVISE granularity (now worked around)
6. **kern.vm_page_speculative_q_age_ms = 500** — Speculative pages evict in 500ms (~1.75 layers at 3.5 t/s)

## Discovered Capabilities (Not Yet Exploited)

| Capability | Source | Potential |
|-----------|--------|-----------|
| MTLIOCommandQueue | Metal framework SPI | Replace pread() pool entirely, uncached file handles, GPU-side IO-compute sync |
| MTL4 ML Command Encoder | Metal 4 framework | Hardware-accelerated ML dispatch, int4/bfloat tensor types |
| MTL4 Compute Substreams | Metal 4 framework | IO/compute overlap within single command buffer |
| MXU (Matrix Extension Unit) | AGX G16X driver | simdgroup_matrix_multiply_accumulate for batch prefill |
| FP8 hardware support | AGX G16X driver | Not useful for IQ2_XXS but could enable new quant formats |
| Morton-order threadgroup walk | MPP programming guide | Better L2 cache reuse for grouped matmul |
| Indirect command buffers | Metal Shading Language spec | GPU-driven MoE expert dispatch without CPU round-trip |
| AGX_DATA_BUFFER_CACHING_OVERRIDE env | AGX driver strings | Force GPU-side cache policy per buffer class |

## Key RE Findings

### From AGX G16X kext analysis:
- GPU priority via `setGPUPriority:` / `AGX_FORCE_GPU_PRIORITY` env var
- Compute encoder coalescing: `tryCoalescingPreviousComputeCommandEncoder`
- DORA (Dynamic Operand Resource Allocation) — G16X-specific resource management
- Spill config registers: CR_PTC_MIN_COUNT, CR_DUPM_MIN_COUNT, CR_MAX_TGP_USAGE

### From APFS/NVMe analysis:
- NVMe queue depth: 128-256 entries (not saturated by 18-thread pool)
- Max single read: 1-2MB before kernel splits
- NVME_PAGE_SIZE = 4096 (alignment handled by kernel automatically)
- speculative_prefetch_max = 192MB total budget (sufficient for our workload)

### From Metal framework analysis:
- MTLCommandQueueDescriptorInternal: enableLowLatencySignalSharedEvent, disableCrossQueueHazardTracking, disableIOFencing, qosLevel, commitsWithQoS
- MTLIOCommandQueue: loadBuffer:offset:size:sourceHandle:sourceHandleOffset:, encodeSignalEvent:value:, priority settings
- Uncached IO file handles bypass page cache at VFS level
