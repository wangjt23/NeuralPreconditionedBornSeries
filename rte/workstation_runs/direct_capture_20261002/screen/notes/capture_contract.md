# Direct segmented TSA capture

The frozen context-segmented run passed all 52 entry checks and 48 full-solve
bitwise comparisons. Its primary cost gate failed at N256: 3.989% improvement
was below 5%; N128 improved 14.18%. This revision changes capture setup only.

API: `DirectSegmentedTSA(base, prototype, chunk=1, capture_first=False)`.
Stage it beside unchanged `segmented_tsa.py` (SHA256
`a95ed61b4c282353e48b8dad6f5aebd993a50f130f10d8aaca6e8aa2c883ba29`).
Only the original TSA base, CUDA complex64 prototype, and this primary mode
are supported. Native complex128 calls retain the original eager fallback.

The parent verified installed FWI Torch2.5.1 context source at
`receipts/torch_graph_context_fwi.txt`. It matches release source: context
entry synchronizes, collects garbage and clears the allocator cache. Their
individual timing contribution remains unmeasured.
[Release source](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/cuda/graphs.py#L153-L168)

The constructor retains one nondefault stream on the prototype device. The
side stream waits for the caller and warms the original stages three times;
the caller waits and a device synchronization completes warmup. On the same
side stream, `capture_begin(capture_error_mode='global')`, the unchanged
subsequent stage, and `capture_end()` create one private graph. No pool hint
is supplied. The caller waits again and a second synchronization completes
setup. Begin/end remain inside the same stream context. End is attempted on
a body error only after begin succeeded. Any capture failure aborts the owned
run; uncertain capture state has no eager fallback.
[Capture implementation](https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/cuda/CUDAGraph.cpp#L76-L147)

The unchanged helper owns replay: input copy, eager original first stage
including zero addition, state copy, then d-1 replays and state copies. Final
complex128 conversion creates fresh owned storage. Every original FP32
operation, cache/layout guard, and base depth remains unchanged. Counts are
4d-1 FFTs and 4d-1 inverse FFTs (63+63 at depth16); one captured stage is 4+4.
Graph, stream, coefficient/cache tensors and captured buffers remain live.
Calls are serial. No GC, allocator or persistent CUDA setting is modified.

CPU tests cannot establish actual GPU capture parity or speed; retain all
parent parity/ownership/guard and complete-cost gates. Skipping cache clearing
may retain ordinary reservations alongside private graph memory. Record
allocated/reserved memory and fail closed on OOM; no shared pools or concurrent
calls are supported.
[Graph memory documentation](https://github.com/pytorch/pytorch/blob/v2.5.1/docs/source/notes/cuda.rst#L1123-L1146)

This implementation task performs no FWI or GPU execution.
