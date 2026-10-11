# GLM-5.3-Flash (glm53): building and fixtures

The glm53 kernels (zig/kernels/cuda/glm53/: kda.cu, hc.cu, exl3.cu) must be built with nvcc 13.3. The Mia
fork's image built its torch extensions with that release. Built with 13.3, every kernel is instruction-identical
to the image's extensions (`sass_check.py`). The node's /usr/local/cuda nvcc 13.0 compiles other SASS for every
kda and hc kernel.

Pass it with `-Dglm53-nvcc`. The other families keep `-Dnvcc`:

    zig build -Dnvcc=/usr/local/cuda/bin/nvcc -Dglm53-nvcc=$HOME/tools/cuda-13.3-tfdsv41/bin/nvcc -Dsm=121 \
      -Doptimize=safe -j8 install native

The build stops with an error when the glm53 kernels' compiler is not release 13.3 (zig/build/cuda.zig
`nvcc_release`). Without `-Dglm53-nvcc` they take `-Dnvcc`, so a 13.0 `-Dnvcc` alone fails.

To check a fresh build against a capture's extensions:

    python3 tools/zig/glm53/sass_check.py --cuobjdump $HOME/tools/cuda-13.3-tfdsv41/bin/cuobjdump \
      ~/glm53fx/kda/extensions/tensorfold_glm_kda_v2.so zig-out/fatbin/glm53_kda.fatbin

The fixtures (`capture.sh <kda|dsa|moe|hc> <work-dir>`) and their digests are listed in FIXTURES.md. The
tests that read them are `tf-cuda-test glm53-kda|glm53-moe <work-dir> <pack-dir> [isolated]` and
`tf-cuda-test glm53-prompt <work-dir> <pack-dir>` (the prompt path over layers 0 and 3 and the MTP layer).

GLM-5.3-Flash-2's reference keeps the DSA caches in FP8 (the Mia fork's TF_GLM_KV=fp8, glm5_next/cuda/kv8.py). Its
fixtures come from the same captures with `KV=fp8 bash capture.sh <dsa|prompt> <work-dir>`, and the tests
`tf-cuda-test glm53-dsa-fp8` and `glm53-prompt-fp8` check them. The engine selects the format with `cuda_glue.Kv`.
Only the kernels that write or read the latent rows and the pooled keys take another Triton variant: `_lat_write`,
`_pool_keys`, `_dense_chunks`, `_sparse_chunks`, `_sparse_onepass`, `_scores_rows`, `_scores`, and the multi-stream
`_seg_*` kernels.
