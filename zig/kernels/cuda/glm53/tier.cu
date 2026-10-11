// Ours: the tiered latent cache's remap (zig/src/families/glm53/kv_tier.zig). A selection's positions become row
// indices at the slab's base: position p of a stream-layer whose map names slot m[p / 4] for its pool reads row
// m[p / 4] * 4 + p % 4. Only the first counts[r] entries of row r change; the attention kernels read the same bytes
// at the new indices as at the old ones in a whole cache.

#include <cstdint>

extern "C" __global__ void tf_glm53_tier_remap(int32_t *tokens, const int32_t *counts, const int32_t *map, int w) {
    const int r = blockIdx.x;
    const int n = counts[r];
    int32_t *row = tokens + (int64_t)r * w;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const int32_t p = row[i];
        row[i] = map[p >> 2] * 4 + (p & 3);
    }
}
