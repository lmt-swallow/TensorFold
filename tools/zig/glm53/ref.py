"""The GLM-5.3-Flash reference (the Mia fork's glm5_next/cuda in its strict setting) driven layer by layer on ONE
Spark for the Zig engine's fixtures. Imported by the dump_*.py scripts before anything of TensorFold:

- STRICT: the environment the recipe's start.sh gives every rank with DENSE=bf16 KV=bf16 KDA_CHUNKED=0
  DRAFTER=mtp (scripts/config.sh of the v1.5 recipe, the image's patches 9f73cca659a1), set here before the
  engine's modules read their switches at import;
- load_rank: one rank of a two-rank split with only the chosen layers' tensors read (weights.load itself, its layer
  list cut down), so a fixture needs a few GiB, not the 82 GiB a rank holds;
- Pair: both ranks in one process, one thread each, taking turns; their all-gathers meet in host memory instead of
  NCCL. An all-gather moves bytes only, so every rank computes the bits it computes on its own Spark;
- Tracer: every call of the wrapped engine functions in order, a tensor argument before the call, after it when the
  call wrote it, and the returned tensors, per rank (r0_ / r1_ prefixes).
"""

from __future__ import annotations

import builtins
import functools
import hashlib
import inspect
import os
import sys
import threading
from pathlib import Path

STRICT = {
    # the strict setting: BF16 dense weights and KV cache, the serial KDA prompt kernel
    "TF_GLM_DENSE": "bf16", "TF_GLM_KV": "bf16", "TF_GLM_KDA_CHUNKED": "0", "TF_GLM_LATENT": "1",
    # the rest of what config.sh exports with DRAFTER=mtp (PARALLEL=1, CONTEXT=524288); none changes a bit by the
    # recipe's own account, but each picks kernels, so the fixtures run what the served ranks run
    "TF_GLM_COMM": "roce", "TF_ROCE_MAX_KB": "512", "TF_GLM_COPY_DRAFTS": "1", "TF_GLM_COPY_MAX": "15",
    "TF_GLM_DFLASH_POLICY": "fnc7:0.3", "TF_GLM_HC_SPLIT": "1", "TF_GLM_PREFILL_OVERLAP": "2",
    "TF_GLM_WIDE_GRAPHS": "16", "TF_GLM_COPY_REPLY_MATCH": "16", "TF_GLM_MULTI_LONE": "0",
    "TF_GLM_MULTI_WINDOW": "32", "TF_GLM_CACHE_ENTRIES": "32", "TF_GLM_CLEAR_THINKING": "0",
    "TF_GLM_MULTI_PREFILL": "1", "TF_GLM_L2PF": "1", "TF_GLM_EXL3_LOADS": "nc", "TF_GLM_SHARED_PREFIX": "1",
    "TF_GLM_MTP": "auto",
}
for _k, _v in STRICT.items():
    os.environ[_k] = _v

import torch  # noqa: E402

DEV = torch.device("cuda:0")
KEEP_IN_BYTES = 64 << 20      # tensor arguments larger than this are weights or whole caches: not saved


def tools_on_path(tools: str) -> None:
    """dsv41's capture helpers (the launch recorder, the fixture writer) beside this folder's."""
    for p in (str(Path(tools) / "dsv41"), str(Path(tools) / "glm53")):
        if p not in sys.path:
            sys.path.insert(0, p)


def bits(t: torch.Tensor) -> torch.Tensor:
    """A tensor's bytes (NaNs and signed zeros compare as bits)."""
    return t.detach().contiguous().view(-1).view(torch.uint8)


def same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(bits(a), bits(b))


# -- one rank's chosen layers ------------------------------------------------------------------------------------
def load_rank(model: str, rank: int, layers: list[int], *, mtp: bool, long_context: bool):
    """weights.load for ``rank`` of two with only backbone ``layers`` (and the MTP layer with ``mtp``) read: the
    loader's own assembly (stacking, splits, absorb matrices, EXL3 words), its layer list cut down. The head, the
    embedding and the final norm load as always (the MTP head's logits and the layers' inputs need them)."""

    from tensorfold.families.glm5_next.cuda import split as split_mod
    from tensorfold.families.glm5_next.cuda import weights as W

    cfg = W.Config.read(model)
    keep = sorted(set(layers))
    if any(not 0 <= i < cfg.layers for i in keep):
        raise ValueError(f"layers {keep} outside 0..{cfg.layers - 1}")

    def cut_range(*a):                    # load()'s `range(cfg.layers)`: the chosen layers; every other range as is
        return list(keep) if a == (cfg.layers,) else builtins.range(*a)

    wanted = {cfg.layers} if mtp else set()
    wanted.update(keep)
    prefetch = split_mod.RankReader.prefetch

    def own_prefetch(self, names, device=None):   # read ahead only what the chosen layers take
        from_layer = [n for n in names if _layer_of(n) in wanted]
        return prefetch(self, from_layer, device) if from_layer else None

    W.range = cut_range
    split_mod.RankReader.prefetch = own_prefetch
    try:
        w = W.load(model, rank=rank, world=2, device="cuda", mtp=mtp)
    finally:
        del W.range
        split_mod.RankReader.prefetch = prefetch
    if [l.index for l in w.layers] != keep:
        raise SystemExit(f"rank {rank}: loaded layers {[l.index for l in w.layers]}, wanted {keep}")
    w.meta["long_context"] = long_context
    return w


def _layer_of(name: str) -> int | None:
    p = "model.language_model.layers."
    if not name.startswith(p):
        return None
    head = name[len(p):].split(".", 1)[0]
    return int(head) if head.isdigit() else None


def weight_shas(w) -> dict[str, str]:
    """sha256 of every tensor a rank's loaded layers hold, by dotted path (the Zig loader's byte-for-byte target)."""

    out: dict[str, str] = {}

    def walk(prefix: str, v) -> None:
        if isinstance(v, torch.Tensor):
            out[prefix] = hashlib.sha256(v.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
        elif isinstance(v, (list, tuple)):
            for i, x in enumerate(v):
                walk(f"{prefix}.{i}", x)
        elif hasattr(v, "__dict__") and not isinstance(v, (str, bytes)):
            for k, x in vars(v).items():
                if not k.startswith("_"):
                    walk(f"{prefix}.{k}", x)

    for l in w.layers:
        walk(f"layers.{l.index}", l)
    if w.mtp is not None:
        walk("mtp", w.mtp)
    walk("norm", w.norm)
    walk("embed", w.embed)
    walk("head", w.head)
    if w.draft_head is not None:
        walk("draft_head", w.draft_head)
    return out


# -- two ranks in one process ------------------------------------------------------------------------------------
class Pair:
    """Two ranks as threads taking turns on one GPU: a rank runs until it needs the other's part of an all-gather,
    then hands over. Only one runs at a time, so the kernels go to the device in a fixed order. ``recorder``: the
    Triton launch recorder, whose current phase each rank keeps across its waits."""

    def __init__(self, recorder=None) -> None:
        self.recorder = recorder
        self.cv = threading.Condition()
        self.turn = 0
        self.parts: list[dict[int, torch.Tensor]] = [{}, {}]
        self.count = [0, 0]
        self.done = [False, False]
        self.error: BaseException | None = None

    def comm(self, rank: int) -> "RankComm":
        return RankComm(self, rank)

    def _wait(self, rank: int, ready) -> None:
        while not ready():
            if self.error is not None:
                raise RuntimeError("the other rank failed") from self.error
            if self.done[1 - rank]:
                raise RuntimeError(f"rank {rank} waits on rank {1 - rank}, which has finished: the ranks' calls differ")
            self.turn = 1 - rank
            self.cv.notify_all()
            rec = self.recorder
            saved = (rec.phase, rec.detail) if rec is not None else None
            self.cv.wait_for(lambda: self.turn == rank or self.error is not None or self.done[1 - rank])
            if rec is not None:
                rec.phase, rec.detail = saved

    def run(self, fn) -> list:
        """``fn(rank)`` on both ranks; their results in rank order."""

        results: list = [None, None]

        def body(rank: int) -> None:
            try:
                with self.cv:
                    self.cv.wait_for(lambda: self.turn == rank or self.error is not None)
                if self.error is None:
                    results[rank] = fn(rank)
                    torch.cuda.synchronize()
            except BaseException as e:  # noqa: BLE001
                self.error = e
            finally:
                with self.cv:
                    self.done[rank] = True
                    self.turn = 1 - rank
                    self.cv.notify_all()

        threads = [threading.Thread(target=body, args=(r,), name=f"rank{r}") for r in (0, 1)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if self.error is not None:
            raise self.error
        return results


class RankComm:
    """``tensorfold.cuda.comm.NCCL``'s all_gather and barrier for one rank of a ``Pair``."""

    world = 2

    def __init__(self, pair: Pair, rank: int) -> None:
        self.p, self.rank = pair, rank

    def all_gather(self, send: torch.Tensor, recv: torch.Tensor) -> None:
        p, r = self.p, self.rank
        k = p.count[r]
        p.count[r] += 1
        torch.cuda.synchronize()
        mine = send.detach().reshape(-1).clone()
        with p.cv:
            p.parts[r][k] = mine
            p._wait(r, lambda: k in p.parts[1 - r])
            other = p.parts[1 - r].pop(k)
        if other.numel() != mine.numel() or other.dtype != mine.dtype:
            raise RuntimeError(f"all-gather {k}: rank {r} sends {mine.numel()} {mine.dtype}, the other "
                               f"{other.numel()} {other.dtype}")
        out = recv.view(2, -1)
        out[r].copy_(mine)
        out[1 - r].copy_(other)

    def barrier(self) -> None:
        self.all_gather(torch.zeros(1, device=DEV), torch.empty(2, device=DEV))


# -- tracing ---------------------------------------------------------------------------------------------------
_local = threading.local()


def _plain(v):
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, (tuple, list)):
        return [_plain(x) for x in v]
    return repr(type(v).__name__)


class Tracer:
    """Records the wrapped functions' tensor arguments around each call while the calling rank's ``step`` is set."""

    def __init__(self) -> None:
        self.arrays: dict[str, torch.Tensor] = {}
        self.calls: list[dict] = []

    @staticmethod
    def begin(rank: int, step: str | None) -> None:
        _local.rank, _local.step, _local.seq = rank, step, 0

    def wrap(self, module, name: str) -> None:
        fn = getattr(module, name)
        if getattr(fn, "_glm53_traced", False):
            return
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            sig = None
        tracer = self

        @functools.wraps(fn)
        def traced(*args, **kwargs):
            step = getattr(_local, "step", None)
            if step is None:
                return fn(*args, **kwargs)
            bound = None
            if sig is not None:
                try:
                    bound = sig.bind(*args, **kwargs)
                except TypeError:
                    bound = None
            items = (list(bound.arguments.items()) if bound else
                     [(f"a{i}", a) for i, a in enumerate(args)] + list(kwargs.items()))
            seq = _local.seq
            _local.seq += 1
            prefix = f"r{_local.rank}_{step}_{seq:03d}_{name}_"
            meta = {"rank": _local.rank, "step": step, "seq": seq, "fn": f"{module.__name__}.{name}", "args": {}}
            before = {}
            torch.cuda.synchronize()
            for k, v in items:
                if isinstance(v, torch.Tensor):
                    nbytes = v.numel() * v.element_size()
                    meta["args"][k] = {"dtype": str(v.dtype), "shape": list(v.shape), "stride": list(v.stride()),
                                       "bytes": nbytes}
                    if nbytes <= KEEP_IN_BYTES:
                        before[k] = v.detach().clone()
                else:
                    meta["args"][k] = _plain(v)
            out = fn(*args, **kwargs)
            torch.cuda.synchronize()
            for k, v in items:
                if isinstance(v, torch.Tensor) and k in before:
                    tracer.arrays[prefix + k + "_in"] = before[k]
                    if not same(v, before[k]):
                        tracer.arrays[prefix + k + "_out"] = v.detach().clone()
                        meta["args"][k]["changed"] = True
            if isinstance(out, torch.Tensor):
                tracer.arrays[prefix + "ret"] = out.detach().clone()
                meta["ret"] = {"dtype": str(out.dtype), "shape": list(out.shape)}
            elif isinstance(out, tuple):
                meta["ret"] = [_plain(x) if not isinstance(x, torch.Tensor) else "tensor" for x in out]
                for j, x in enumerate(out):
                    if isinstance(x, torch.Tensor):
                        tracer.arrays[prefix + f"ret{j}"] = x.detach().clone()
            tracer.calls.append(meta)
            return out

        traced._glm53_traced = True
        setattr(module, name, traced)

    def wrap_all(self, spec) -> None:
        for module, names in spec:
            for n in names:
                self.wrap(module, n)


def seeded(seed: int) -> torch.Generator:
    g = torch.Generator(device=DEV)
    g.manual_seed(seed)
    return g


def ids_for(model: str, n: int, seed: int) -> list[int]:
    """``n`` token ids: a fixed English text's (repeated), then seeded ids past it; the same on every run."""

    from tokenizers import Tokenizer

    text = ("GLM-5.3-Flash interleaves three linear-attention (KDA) layers with one sparse-attention (DSA) layer. "
            "Explain how a draft window that is partly rejected must leave the recurrent state exactly as the serial "
            "steps leave it, and why the sparse layer's pooled keys must not depend on the window size. ")
    tok = Tokenizer.from_file(str(Path(model) / "tokenizer.json"))
    base = tok.encode(text).ids
    out = (base * (n // max(1, len(base)) + 1))[:min(n, 4 * len(base))]
    g = torch.Generator()
    g.manual_seed(seed)
    if len(out) < n:
        out += torch.randint(1000, 150000, (n - len(out),), generator=g).tolist()
    return out


def provenance(out: Path, extra: dict) -> None:
    import json

    import triton
    import tensorfold

    (out / "capture.json").write_text(json.dumps({
        "torch": torch.__version__, "triton": triton.__version__, "tensorfold": tensorfold.__version__,
        "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(),
        "image_patches": os.environ.get("TF_GLM53_IMAGE_PATCHES", ""), "strict": STRICT, **extra},
        indent=1, default=str) + "\n")
