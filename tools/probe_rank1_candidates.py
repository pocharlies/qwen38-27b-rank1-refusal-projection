#!/usr/bin/env python3
"""Is a published "abliterated" checkpoint actually a rank-1 directional edit?

    python3 tools/probe_rank1_candidates.py                      # the three from §2
    python3 tools/probe_rank1_candidates.py --abl OTHER/REPO     # any candidate
    python3 tools/probe_rank1_candidates.py --abl A --abl B --json out.json

Answers it for ~430 MB per candidate instead of a 54 GB download, by reading
individual tensors out of the remote safetensors over HTTP range requests: 8 bytes
of header length, then the JSON header, then exactly the byte range of the tensor
you asked for. Nothing else crosses the wire.

WHY THIS EXISTS. "Abliterated" in a repo name tells you nothing about *how* the
model was edited. Of the three published Qwen3.8-27B ablations measured for §2,
only one was a clean rank-1 directional edit; the other two are projection-shaped
or plain fine-tunes, and the runtime dial cannot reproduce them at any λ. Measuring
that ahead of time costs minutes. Finding out afterwards costs a full extraction
run and a wrong conclusion.

WHAT IT REPORTS, per module:

  s0/s1        top singular value of ΔW over the second. Near 1 means NO dominant
               direction exists — whatever that repo did, it was not a directional
               ablation. This is the gate that rejected `trohrbaugh` (1.14–4.59).
  rank1E       fraction of ΔW's energy in the first singular component.
  cos          |cos(v0, u0ᵀW_base)|. Asks whether ΔW even has the *shape* of a
               projection −λ·r̂r̂ᵀW. This is what rejected `trohrbaugh` (0.17–0.46).
  λ_eff        how many times the edit removes the direction. ≈1.0 is a clean single
               removal. Overshoot is a real failure mode and it is not rare:
               Blackfrost-AI/…-ABLITERATED-BF16 measures 2.599 — past the DeepSeek-V4
               regime (2.43) where the component is not removed but *inverted*.
  coherence    |cos| between the dominant directions of DIFFERENT modules. High means
               one global direction (Ektome ≥0.9995); low means the edit is per-layer
               and there is no single axis to extract.

READ s0/s1 BEFORE cos. When s0 ≈ s1 the "dominant direction" u0 is arbitrary, so its
cos against anything is noise dressed as a measurement. A candidate with cos = 0.99
and s0/s1 = 1.2 is not projection-shaped; it is undetermined.

Baseline for "is this cos meaningful": two random unit vectors in R^5120 have
|cos| ≈ 1/sqrt(5120) = 0.0140. Anything at that level is orthogonal, not related.

Pin revisions with --rev when you publish a number. HuggingFace `main` moves, and a
measurement against a moving target is not reproducible.
"""

from __future__ import annotations

import argparse
import functools
import json
import struct
import sys
import urllib.request

import numpy as np

UA = {"User-Agent": "qwen38-rank1-probe/1.0"}

BASE_DEFAULT = "Qwen/Qwen3.8-27B"
# The three measured for §2 of the README. The verdicts there came from this code.
CANDIDATES_DEFAULT = [
    "Zynerji/Ektome-Qwen3.8-27B-PristinelyUncensored",
    "trohrbaugh/Qwen3.8-27B-heretic-ara",
    "orwelian84/Qwen3.8-27B-OBLITERATUS-Advanced",
]

# Modules that WRITE TO THE RESIDUAL STREAM — the only ones where a refusal direction
# can live. Layer 20 is a linear-attention layer, 31/47/63 are full-attention
# (`full_attention_interval: 4`, so layers 3, 7, … 63). Sampling both kinds and
# spreading across depth catches per-layer edits that a single probe would miss.
MODULES_DEFAULT = [
    "model.language_model.layers.20.linear_attn.out_proj.weight",
    "model.language_model.layers.31.self_attn.o_proj.weight",
    "model.language_model.layers.47.self_attn.o_proj.weight",
    "model.language_model.layers.63.mlp.down_proj.weight",
]

RANDOM_COS_5120 = 1.0 / np.sqrt(5120)  # 0.01398


def _get(url: str, rng: tuple[int, int] | None = None) -> bytes:
    h = dict(UA)
    if rng:
        h["Range"] = f"bytes={rng[0]}-{rng[1]}"
    req = urllib.request.Request(url, headers=h)
    with urllib.request.urlopen(req, timeout=180) as r:
        return r.read()


def _resolve(repo: str, fname: str, rev: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/{rev}/{fname}"


@functools.lru_cache(maxsize=64)
def _index(repo: str, rev: str):
    """tensor -> shard. Returns None for single-shard repos (no index file)."""
    try:
        raw = _get(_resolve(repo, "model.safetensors.index.json", rev))
        return json.loads(raw)["weight_map"]
    except Exception:  # noqa: BLE001
        return None


@functools.lru_cache(maxsize=256)
def _header(repo: str, shard: str, rev: str):
    """The safetensors header: u64 length, then that many bytes of JSON. A few KB."""
    url = _resolve(repo, shard, rev)
    n = struct.unpack("<Q", _get(url, (0, 7)))[0]
    return json.loads(_get(url, (8, 8 + n - 1))), 8 + n


def tensor(repo: str, key: str, rev: str = "main") -> np.ndarray:
    """One tensor as float64 [out, in]. Only that byte range is fetched."""
    wm = _index(repo, rev)
    shard = "model.safetensors" if wm is None else wm.get(key)
    if shard is None:
        raise KeyError(f"{key} not in {repo}")
    hdr, start = _header(repo, shard, rev)
    if key not in hdr:
        raise KeyError(f"{key} not in shard {shard} of {repo}")
    meta = hdr[key]
    a, b = meta["data_offsets"]
    raw = np.frombuffer(_get(_resolve(repo, shard, rev), (start + a, start + b - 1)), dtype=np.uint8)
    dt = meta["dtype"]
    if dt == "BF16":
        # numpy has no bfloat16: widen to float32 by placing the 16 bits in the high half.
        arr = (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    elif dt == "F32":
        arr = raw.view(np.float32).astype(np.float64)
    elif dt == "F16":
        arr = raw.view(np.float16).astype(np.float64)
    else:
        raise RuntimeError(f"dtype {dt} unsupported by this reader")
    return arr.reshape(meta["shape"])


def top_singular(dW: np.ndarray, iters: int = 200):
    """(u0, v0, s0, s1, rank1_energy) by power iteration + deflation.

    Full SVD of a [5120, 5120] float64 matrix is wasteful when only the top two
    singular values matter, and this is the same method the extractor uses — so the
    numbers here are comparable with what actually ships in the directions file.
    """
    rng = np.random.default_rng(0)  # fixed: two runs on the same weights must agree
    v = rng.standard_normal(dW.shape[1])
    v /= np.linalg.norm(v)
    for _ in range(iters):
        u = dW @ v
        nu = np.linalg.norm(u)
        if nu < 1e-30:
            return None
        u /= nu
        v = dW.T @ u
        nv = np.linalg.norm(v)
        if nv < 1e-30:
            return None
        v /= nv
    s0 = float(u @ dW @ v)
    dW1 = dW - s0 * np.outer(u, v)
    v2 = rng.standard_normal(dW.shape[1])
    v2 /= np.linalg.norm(v2)
    for _ in range(iters):
        u2 = dW1 @ v2
        n2 = np.linalg.norm(u2)
        if n2 < 1e-30:
            break
        u2 /= n2
        v2 = dW1.T @ u2
        n2 = np.linalg.norm(v2)
        if n2 < 1e-30:
            break
        v2 /= n2
    s1 = float(abs(u2 @ dW1 @ v2)) if np.linalg.norm(dW1) > 1e-30 else 0.0
    fro2 = float((dW ** 2).sum())
    return u, v, s0, s1, (s0 * s0 / fro2 if fro2 > 0 else 0.0)


def probe_module(base_repo, abl_repo, key, base_rev, abl_rev):
    Wb = tensor(base_repo, key, base_rev)
    Wa = tensor(abl_repo, key, abl_rev)
    if Wb.shape != Wa.shape:
        return {"key": key, "error": f"shape {Wb.shape} vs {Wa.shape}"}
    dW = Wa - Wb
    delta = float(np.linalg.norm(dW) / max(np.linalg.norm(Wb), 1e-30))
    if np.linalg.norm(dW) < 1e-12:
        # Byte-identical. Not "no ablation found" — this module was simply not touched,
        # which is itself a fact worth printing (see README on the vision tower).
        return {"key": key, "delta": 0.0, "untouched": True}
    got = top_singular(dW)
    if got is None:
        return {"key": key, "delta": delta, "error": "power iteration did not converge"}
    u0, v0, s0, s1, rank1E = got
    # Does ΔW look like −λ·r̂r̂ᵀW? Then v0 should align with u0ᵀW_base.
    w = Wb.T @ u0
    nw = np.linalg.norm(w)
    cos = float(abs(v0 @ (w / nw))) if nw > 1e-30 else float("nan")
    # λ_eff: how many times over the edit removes the component. Sign matters —
    # a positive s0 with this construction would mean the edit ADDS the direction.
    denom = float(u0 @ Wb @ v0)
    lam_eff = float(-s0 / denom) if abs(denom) > 1e-30 else float("nan")
    return {
        "key": key, "delta": delta, "untouched": False,
        "s0": s0, "s1": s1, "s0_over_s1": (s0 / s1 if s1 > 1e-30 else float("inf")),
        "rank1E": rank1E, "cos": cos, "lam_eff": lam_eff,
        "u0": u0.tolist() if False else None,  # not serialised: 5120 floats per module
        "_u0": u0,
    }


def verdict(rows):
    """A candidate is usable only if EVERY probed module is a clean rank-1 edit."""
    ok = [r for r in rows if not r.get("error") and not r.get("untouched")]
    if not ok:
        return "NO DATA"
    if min(r["s0_over_s1"] for r in ok) < 5:
        return "REJECTED (no dominant direction)"
    if min(r["cos"] for r in ok) < 0.9:
        return "REJECTED (not projection-shaped)"
    if max(r["lam_eff"] for r in ok) > 2.0:
        return "CAUTION (over-ablated: inverts the component)"
    return "CLEAN rank-1"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=BASE_DEFAULT)
    ap.add_argument("--base-rev", default="main", help="pin this when publishing a number")
    ap.add_argument("--abl", action="append", default=None, help="repeatable; default = the three from §2")
    ap.add_argument("--rev", action="append", default=None, help="revision per --abl, in order")
    ap.add_argument("--module", action="append", default=None, help="repeatable; default = 4 residual writers")
    ap.add_argument("--json", default=None, help="write full results here")
    args = ap.parse_args()

    cands = args.abl or CANDIDATES_DEFAULT
    revs = args.rev or []
    mods = args.module or MODULES_DEFAULT
    out = {"base": args.base, "base_rev": args.base_rev, "candidates": {}}

    print(f"base: {args.base}@{args.base_rev}")
    print(f"random-|cos| baseline in R^5120: {RANDOM_COS_5120:.5f}\n")

    for i, abl in enumerate(cands):
        rev = revs[i] if i < len(revs) else "main"
        print(f"=== {abl}@{rev}")
        rows = []
        for key in mods:
            try:
                r = probe_module(args.base, abl, key, args.base_rev, rev)
            except Exception as e:  # noqa: BLE001
                r = {"key": key, "error": str(e)}
            rows.append(r)
            short = key.replace("model.language_model.layers.", "L").replace(".weight", "")
            if r.get("error"):
                print(f"  {short:<34} ERROR: {r['error']}")
            elif r.get("untouched"):
                print(f"  {short:<34} byte-identical to base (not edited)")
            else:
                print(f"  {short:<34} s0/s1={r['s0_over_s1']:8.2f}  rank1E={r['rank1E']:.4f}  "
                      f"cos={r['cos']:.4f}  lam_eff={r['lam_eff']:.4f}  delta={r['delta']:.5f}")

        # Coherence: is this ONE global direction, or a different edit per layer?
        us = [r["_u0"] for r in rows if r.get("_u0") is not None]
        coh = None
        if len(us) >= 2:
            cs = [abs(float(us[a] @ us[b])) for a in range(len(us)) for b in range(a + 1, len(us))]
            coh = {"min": min(cs), "median": float(np.median(cs))}
            print(f"  {'coherence between modules':<34} min={coh['min']:.4f}  median={coh['median']:.4f}"
                  f"   -> {'ONE global axis' if coh['min'] > 0.9 else 'per-layer, no single axis'}")
        v = verdict(rows)
        print(f"  VERDICT: {v}\n")
        for r in rows:
            r.pop("_u0", None)
            r.pop("u0", None)
        out["candidates"][abl] = {"rev": rev, "modules": rows, "coherence": coh, "verdict": v}

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"written: {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
