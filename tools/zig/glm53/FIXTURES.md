# GLM-5.3-Flash fixtures

These are the fixtures the Zig tests read. A recapture with `capture.sh` (same image, pack and scripts) should
reproduce these digests. If one differs, compare the per-file lists first.

Each fixture was captured with `bash tools/zig/glm53/capture.sh <name> <work-dir>` from this tree. They were captured
on spark-9e43 (GB10, CUDA 13.3, torch 2.13.0a0, the recipe's image `sha256:ef83797d…`, tf.patches 9f73cca659a1),
from the pack Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold at snapshot
078455ffe6472f9a52fbc1139f58b9db2881b25c. On that node they sit in `~/glm53fx/<name>/` (dsa in `~/glm53fx/dsa2/`: the capture that also saves run l's drawn caches).

`tree` is the sha256 of `find out aot -type f | sort | xargs sha256sum`, run inside the work dir. `manifest` is the
sha256 of `out/fixture/manifest.json`, which holds the params, the array names, dtypes and shapes, and the call trace.
The per-file lists, including `extensions/`, are on the node in `~/glm53fx/<name>.sha256`.

| name | tree | manifest | files | bytes |
| --- | --- | --- | ---: | ---: |
| hc | 7389e54884043073412b3692b4c0512bad066c983eee4d56c047219a65700125 | 5c3ad7ba7cd3c80eb580b3fdb05ddb3f31a24690f3a7559daf05105c88c9f6cd | 104 | 433,484,979 |
| kda | d68d0034ddfd24a1c1b5654be1e0bcc71c4f91153e8e71b743737c8529d8108b | 7c6fc9f284a1ceee6e501e3b8b8ea40651751728ee9d20d6cedf7f9b4e87ad4e | 2,212 | 5,127,428,621 |
| moe | 2464dd3ab5ad1f49241bad1fb0e9903a799b1d50d779d907520e7d0704ee43b5 | 5ab543c75cae01a45a44176f61f4818784868699a2abb1be1e43ba9d63d98b74 | 547 | 2,229,724,806 |
| dsa | 0e37decbe600f775db1814efde54b351b5b3af2cd8654b89b53cd0b2df7c74a5 | 0629630da3ee75cb7fea3c428e22acc9a7b1901f14490b0691356b23376df7c6 | 9,352 | 27,323,757,831 |
| prompt | 0405547653a2c6c7d25262173e8e063880ab3a813bfe08f8305fd7437f769259 | e0ace6c62f9bfe92bf9a5f1e5b8d3be9772ca1c6920bf4122b52300f412b05ea | 327 | 43,518,206 |

The Triton kernels each capture launched: hc 8, kda 33, moe 26, dsa 127, prompt 140. The prompt fixture holds
sha256 digests of each traced call's outputs instead of the arrays (dump_prompt.py), hence its size. In every fixture both ranks' streams
were byte-equal at every step.

`tf-cuda-test glm53-kda`, `glm53-moe` and `glm53-dsa` match these fixtures byte for byte, on both ranks: 418, 166 and
1,154 comparisons, plus every loaded tensor's sha256. `glm53-prompt` matches the prompt fixture: 45 chunks, 8,586
digests.
