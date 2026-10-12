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

// The slab's slots in device memory (kv_tier.Slab, cuda_tier.zig): `words` 16-byte words a slot (a pool's 4 rows:
// 256 in bf16, 132 in FP8), a block a slot. scatter: staged slot i (`stride` words apart) into slot slots[i];
// gather: slot slots[i] into dst slot i (`stride` words apart); copy: slot dst[i] := slot src[i] (no two copies
// of a call share a slot).
extern "C" __global__ void tf_glm53_tier_scatter(const uint4 *src, uint4 *slab, const int32_t *slots, int words, int stride) {
    const int i = blockIdx.x;
    for (int w = threadIdx.x; w < words; w += blockDim.x) slab[(int64_t)slots[i] * words + w] = src[(int64_t)i * stride + w];
}

extern "C" __global__ void tf_glm53_tier_gather(const uint4 *slab, uint4 *dst, const int32_t *slots, int words, int stride) {
    const int i = blockIdx.x;
    for (int w = threadIdx.x; w < words; w += blockDim.x) dst[(int64_t)i * stride + w] = slab[(int64_t)slots[i] * words + w];
}

extern "C" __global__ void tf_glm53_tier_copy(uint4 *slab, const int32_t *dst, const int32_t *src, int words) {
    const int i = blockIdx.x;
    for (int w = threadIdx.x; w < words; w += blockDim.x) slab[(int64_t)dst[i] * words + w] = slab[(int64_t)src[i] * words + w];
}
