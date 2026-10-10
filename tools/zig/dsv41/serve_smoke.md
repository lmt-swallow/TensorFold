# DeepSeek-V4.1-Flash on the native server over two DGX Sparks: the smoke

`serve_smoke.sh` runs the steps below end to end (`bash serve_smoke.sh`; `bash serve_smoke.sh stop` removes both
containers).

Rank 0 (spark-616b, 10.0.1.1) serves HTTP; rank 1 (spark-9e43, 10.0.1.2) follows its steps. Both need: the pack
(`~/models/DeepSeek-V4.1-Flash-EXL3-2.9bpw`), their own Python rank cache (`~/tfcache/rank/rank{r}of2-*.bin`,
written by a `tensorfold serve` of the Python fork with `TF_DS_RANK_CACHE`), the captured Triton set with the RoPE
tables beside it (`~/aot/dsv41-all/{aot.json,cubins,rope}`; `tools/zig/dsv41/dump_rope.py --rows 20480` for a
16384-token context: rows = cuda_lanes.ropeRows(context)), and `zig-out/native/bin/tensorfold-native` built on a
GB10 (`zig build -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121 -Doptimize=safe native`), run inside
`nvcr.io/nvidia/pytorch:26.07-py3` (the host has no libnccl).

```
DOCK="--gpus all --ipc=host --net=host --shm-size 16g --device /dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1
      -v $HOME/models:/models:ro -v $HOME/tfcache/rank:/rc:ro -v $HOME/aot/dsv41-all:/k:ro -v $HOME/bin:/b:ro"
ENVS="-e TENSORFOLD_CUDA_KERNELS=/k -e TF_DS_RANK_CACHE=/rc -e TENSORFOLD_MEMORY_RESERVE_GIB=4
      -e TF_DS_ENGRAM=/models/DeepSeek-V4.1-Flash-Engram -e NCCL_IB_HCA=rocep1s0f0 -e NCCL_IB_GID_INDEX=3 -e NCCL_SOCKET_IFNAME=enp1s0f0np0"
SERVE="/b/tensorfold-native serve /models/DeepSeek-V4.1-Flash-EXL3-2.9bpw --tp 2 --master 10.0.1.1 --master-port 29551
      --context 16384 --parallel 1 --temperature 0 --no-thinking --no-drafts --no-update-check --name dsv41
      --host 0.0.0.0 --port 8000"
# rank 0 on spark-616b, then rank 1 on spark-9e43 (rank 1 retries the master port for 30 minutes)
docker run -d --name tfz0 $DOCK $ENVS nvcr.io/nvidia/pytorch:26.07-py3 bash -c "$SERVE --rank 0"
ssh 10.0.1.2 docker run -d --name tfz1 $DOCK $ENVS nvcr.io/nvidia/pytorch:26.07-py3 bash -c "$SERVE --rank 1"
```

Engram (`TF_DS_ENGRAM`, the original FP8 tables beside the pack) needs the Python engine's compressed token map: it
is `~/aot/dsv41-all/token_map.json` on both nodes (`<kernels>/token_map.json`, the default; `TF_DS_TOKEN_MAP`
overrides), written once by the fork's `ops.compressed_token_map` from the pack's tokenizer.json (sha256
de40bcdd...5dc28, 129,280 ids to 99,092 classes). Without `TF_DS_ENGRAM` the ranks skip Engram as the fork does
without tables, and the output is degraded.

`--tp/--rank/--master/--master-port` export `TF_TP_WORLD/RANK/MASTER/PORT`, which the family reads (environment
alone works too). Rank 1's process opens the engine and runs the follower loop inside `open`; it never listens.

Then, from spark-616b:

```
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"dsv41",
  "messages":[{"role":"user","content":"Write a Python function that parses an HTTP request line and returns method, path, version."}],
  "max_tokens":64,"temperature":0}'
```

Reference: the Python fork's greedy reply for the same prompt (`~/bin/home_start_dsv41.sh` starts it with the same
pack on the same pair; stop it with `docker stop dsv41-master` and `ssh 10.0.1.2 docker stop dsv41-worker` before
starting the native ranks: each needs ~100 GiB). Compare token ids (`/v1/tokenize` on either server), not text.

What the interim engine does not do yet: the fork's prompt kernels (prompts run through the decode layer loop in
16-row windows, so a prompt's cache bits can differ from the Python prompt path), sampling above temperature 0,
DSpark drafts, several streams, Engram's gate (no tables are read), prompt snapshots. The context bucket (the
indexer's scan) is the power of two at or above `--context` (`TF_DS_BUCKET` overrides); the captured kernel set
covers buckets the serving capture saw (prompts to 9,000 tokens).

## One model per node

Host and GPU share the Spark's unified memory, so a fixture test or a capture started beside a serving rank does not
fail an allocation: it swaps the node until nothing answers (sshd included), and a two-rank server takes the other
node with it. Run `tools/zig/dsv41/gpu_guard.sh` on the node before any `docker run --gpus` (the capture scripts do;
test containers must too: `ssh node bash -s < tools/zig/dsv41/gpu_guard.sh && ssh node docker run ...`), and run
`serve_smoke.sh stop` as soon as a measurement ends; `KEEP=1` is for serving, not for a pause between tests.
