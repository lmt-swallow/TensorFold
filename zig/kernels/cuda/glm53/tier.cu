// Ours: the tiered latent cache's remap (zig/src/families/glm53/kv_tier.zig). A selection's positions (tokens) become
// row indices at the slab's base (out, the same shape; the positions stay for whoever reads them): position p reads row m[p / 4] * 4 + p % 4, where m is the map of the stream that holds
// p's pool: the stream itself, or for an alias (a kept prompt continued) an ancestor's below its fork (past the home
// block). Only the first counts[r] entries of row r are written; the attention kernels read the same bytes at the new
// indices as at the old ones in a whole cache.

#include <cstdint>

// chain: n, home_pools, fork[8], map[8] (int64 each; kv_tier.max_chain): level 0 is the stream, level i + 1 serves pools in
// [home_pools, fork[i]) of level i.
extern "C" __global__ void tf_glm53_tier_remap(const int32_t *tokens, int32_t *out, const int32_t *counts, const int64_t *chain, int w) {
    const int r = blockIdx.x;
    const int n = counts[r];
    const int64_t levels = chain[0], home = chain[1];
    const int32_t *row = tokens + (int64_t)r * w;
    int32_t *dst = out + (int64_t)r * w;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const int32_t p = row[i];
        const int64_t q = p >> 2;
        int lvl = 0;
        while (lvl + 1 < levels && q >= home && q < chain[2 + lvl]) lvl++;
        const int32_t *map = reinterpret_cast<const int32_t *>(chain[10 + lvl]);
        dst[i] = map[q] * 4 + (p & 3);
    }
}
