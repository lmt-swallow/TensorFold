# DeepSeek-V4.1 CUDA exact cold suffix cone

This opt-in path belongs to the tested CUDA implementation on this branch. It is not a port to the newer split DeepSeek-V4.1 implementation on upstream main.

## Enable

Set the same options on both tensor-parallel ranks:

```sh
TF_DS_EXACT_CONE=1
TF_DS_REPLAY=0
TF_DS_KEEP=0
TENSORFOLD_PREFILL_INTERLEAVE=0
TENSORFOLD_PREFILL_OVERLAP=0
```

The default remains full prefill. Turning `TF_DS_EXACT_CONE` off restores that path. Combining exact cone with bounded replay is rejected. Ineligible admissions use full prefill.

Eligibility requires a cold, isolated admission with no kept prefix, partial prefill or overlapping pass. The model must match the checked 40-layer, two-rank layout, 128-row window, 2,048-row prompt chunks and supported prompt kernels. With DSpark, the checked three-stage head, 144-row rings, five-token block and taps at layers 37–39 are required. Both ranks agree on the admission flag before entering its collectives.

## Computation and state

The lower half still processes every original prompt chunk and produces the compressed pools. The upper window layers process a conservative suffix beginning at an original chunk boundary. Its halo is `144 + 20 * 127 = 2684` rows; for 65,536 input tokens the boundary is 61,440. Absolute positions, original chunk boundaries, arithmetic, weights and cache precision are preserved.

The boundary is fixed on the admission's pass and is independent of prefix-reuse offsets. DSpark absorbs the original suffix taps and retains its terminal rings and absolute absorbed position. This optimization does not add main-model checkpoints.

## Diagnostic gates

`dsv41-prompt-time` contains the isolated full/cone state and fixed-512 continuation checks. `dsv41-runtime-cone` exercises the actual lane backend and core with cold full, cold cone and a released-slot full admission, exports raw input/output IDs and compares every generated token. The synthetic inputs are deterministic and contain no customer records.

Qualification receipts and timing belong in the PR description. A fixed-token parity run does not qualify HTTP/template handling, batching, KEEP, resumed contexts, four retained contexts or every context length. The implementation remains opt-in until those applicable gates are run on the integration target.
