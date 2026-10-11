# GLM-5.3-Flash reference baselines (B1)

The Zig engine must reproduce these replies byte for byte. They come from the Mia recipe v1.5 in the strict setting:
`reference.sh start`, DENSE=bf16 KV=bf16 KDA_CHUNKED=0 DRAFTER=mtp PARALLEL=1 CONTEXT=524288 VISION=0. The image
is pinned by digest (sha256:ef83797d…, tf.patches 9f73cca659a1). Head spark-9e43, worker spark-616b, 2026-10-11.

The run used `bench.py`: greedy, 512 tokens, `return_token_ids`. Each prompt was sent twice with drafts and once
with `"draft": false`. The sha is the first 16 hex digits of the sha256 of the reply's ids as JSON. The results
are in `results/20261011-b1-bench{2,3}.json`; run 3 came after a server restart.

| prompt | prompt tokens | reply sha | rounds | tokens a round | drafted / accepted | decode tok/s |
| --- | ---: | --- | ---: | ---: | --- | ---: |
| code | 52 | 5641a11ca507b521 | 180 | 2.839 | 476 / 332 | 34.9 |
| prose | 49 | c9fdc1c230b4ab85 | 228 | 2.241 | 490 / 283 | 29.4 |
| structured | 50 | ef9f4bb524149b69 | 166 | 3.078 | 458 / 345 | 37.7 |
| needle64k | 63,985 | bf36e4ee1b5ed74f | 32 | 2.969 | 121 / 63 | 33.7 |

The replies were byte-identical across all three runs, with and without drafts. The needle answer was right.
With VISION=1 the code prompt gives the same sha (`results/20261011-b1-vision-code.json`).

Prefill rates, from prompts that share no prefix (nothing cached):

| prompt | tok/s |
| ---: | ---: |
| 32k | 1,604 |
| 64k | 1,584 |
| 128k | 1,534 |

Load: 83-95 s. Each rank holds ~93.3 GiB on the GPU, against a startup estimate of 94.09 GiB.
MemAvailable after the load: head 13.5 GiB, worker 14.7 GiB.

The 45-layer forward fixture comes from `forward_capture.sh`: one served request, a 3,000-token prompt and 64
drafted tokens. The reply sha is 496ee8306bd5f511. The fixture sits in `~/glm53fx/fwd` on both nodes.
- With TF_GLM_HC_SPLIT=0 (`fwd0`), every one of the 438 arrays both captures hold is byte-identical on both ranks.
- The two ranks' streams are equal too.
