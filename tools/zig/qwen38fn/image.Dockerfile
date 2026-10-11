# The bit-exact reference for Qwen3.8-Flash-Next on GB10: upstream TensorFold Python 0.6.6 (cb2ebf05), unpatched,
# on NVIDIA's PyTorch 26.07 container. Build it on the node from a context holding this file and the tagged tree:
#   git -C ~/src/TensorFold archive --format=tar.gz -o <ctx>/tensorfold-cb2ebf05.tar.gz cb2ebf05
#   cp image.Dockerfile <ctx>/Dockerfile && docker build -t tensorfold-qwen38fn:0.6.6 <ctx>
# The CUDA extensions build on first use (they need the GPU's capability): mount a cache at /cache to keep them.
FROM nvcr.io/nvidia/pytorch:26.07-py3
ARG TF_COMMIT=cb2ebf0540f42604e2759b2ddef497861e928248
LABEL tf.version=0.6.6 tf.commit=${TF_COMMIT} org.opencontainers.image.source=https://github.com/ashhart/TensorFold
COPY tensorfold-cb2ebf05.tar.gz /opt/
RUN mkdir -p /opt/tensorfold && tar -xzf /opt/tensorfold-cb2ebf05.tar.gz -C /opt/tensorfold \
 && rm /opt/tensorfold-cb2ebf05.tar.gz \
 && pip install --no-cache-dir "/opt/tensorfold[grammar]" "av>=12" \
 && python -c "import tensorfold, torch; print(tensorfold.__version__, torch.__version__, torch.version.cuda)" \
 && tensorfold --version
ENV TORCH_EXTENSIONS_DIR=/cache/torch_extensions TRITON_CACHE_DIR=/cache/triton
WORKDIR /workspace
ENTRYPOINT ["tensorfold"]
