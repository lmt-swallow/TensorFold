# Qwen3.8-Flash-Next: the 0.6.6 reference, its fixtures and its Triton kernels

The Zig engine's Flash Next family (`zig/src/families/qwen38fn/`) is checked bit for bit against upstream
TensorFold 0.6.6 (ashhart/TensorFold v0.6.6, cb2ebf05) serving turboderp's EXL3 pack `3.05bpw_h5_ng5` on one DGX
Spark with bf16 KV. Its Triton kernels are replayed from cubins captured from that engine (`cuda.aot.Set`), its CUDA
kernels are 0.6.6's sources built with the extensions' nvcc flags (`zig/build/cuda.zig`).

Everything here needs the GPU to itself: host and GPU share one memory, so stop any serving model first and run
`tools/zig/dsv41/gpu_guard.sh` on the node (each script does).

## The image and the pack

`image.Dockerfile` builds `tensorfold-qwen38fn:0.6.6`: NVIDIA's `pytorch:26.07-py3` plus the tagged tree, installed
unpatched (its header has the two commands). The CUDA extensions compile on first use for the GPU present; the
scripts mount `~/.cache/tensorfold-qwen38fn-0.6.6` at `/cache` so they compile once.

The pack lives at `~/models/Qwen3.8-Flash-Next-exl3-3.05bpw_h5_ng5` on spark-616b (85,139,442,313 bytes, 25 files,
each checked against the Hugging Face tree's sha256 / git blob ids; `.hf_tree.json` beside it is that manifest).

## 1. The served reference

```
KEEP=1 bash tools/zig/qwen38fn/reference_serve.sh          # start, smoke (token_sha), keep serving
python3 tools/zig/dsv41/bench_watch.py --host spark-616b --model qwen38fn --only short,concurrent --label tf066-exl3-305
bash tools/zig/qwen38fn/reference_serve.sh stop
```

The server runs `tensorfold serve /model --parallel 4 --context 262144` (bf16 KV, MTP drafts on: 0.6.6's defaults).
The smoke uses the DSV4.1 smoke's prompt (greedy, 64 tokens), so its `token_sha` is the reference's for that request.
`CAPTURE=1` adds the recorder shim (`tools/zig/dsv41/shim/usercustomize.py`), which writes every launched Triton
specialization to `~/aot/qfn-serve-capture/out/rank0/launches.json` with the cubins in `~/aot/qfn-serve-capture/triton`;
touch `out/rank0/dump.now` before stopping, then pack as in `tools/zig/dsv41/serve_capture.md` §3 with
`--capture ~/aot/qfn-serve-capture/out/rank0/launches.json ~/aot/qfn-serve-capture/triton /aot/triton`.

## 2. Layer fixtures

```
bash tools/zig/qwen38fn/capture_layer0.sh ~/aot/qfn-layer0        # dump_layer0.py in the image, then the AOT pack
zig-out/bin/tf-cuda-test qwen38fn-layer0 ~/aot/qfn-layer0 ~/models/Qwen3.8-Flash-Next-exl3-3.05bpw_h5_ng5 [isolated]
```

`dump_layer0.py` loads layer 0 of the pack alone (its output depends on the tokens and its own state only), decodes
a fixed prompt in windows of 1, 2, 3, 5, 7 and 16 rows (two of them with a rejected tail, so the commit replays the
DeltaNet state), and saves every stage: the embedding, both hyper-connections' write-back and read-out, the
DeltaNet projection, chain, replay inputs, state and conv window, the output projection, the router, the picks, the
experts' slots and the MoE write-back. Each window also runs through 0.6.6's own `layer_forward` from the same state;
the script stops unless both give the same bits, so the staging is the engine's own sequence.

The Zig test carries its own state from window to window and compares every stage's bytes. `isolated` feeds each
stage the reference's inputs instead of its own outputs, so one wrong kernel cannot hide the stages after it.
