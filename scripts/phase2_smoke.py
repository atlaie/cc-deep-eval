"""
Phase 2 smoke test — MoE routing capture (Extension 1).

Usage:
  export VLLM_BASE_URL="https://cc-deep-eval.debug.pour-demain.containers.tinfoil.dev/v1"
  export VLLM_API_KEY="<key>"
  python scripts/phase2_smoke.py --n-pairs 3 --routing-layers 3 39 62 --dump-first
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

try:
    from vllm_lens._helpers._serialize import deserialize_tensor
except Exception:
    import base64 as _b64, numpy as _np
    def deserialize_tensor(d):
        import zstandard as _zstd
        raw = _b64.b64decode(d["data"])
        if d.get("compression") == "zstd":
            raw = _zstd.ZstdDecompressor().decompress(raw)
        arr = _np.frombuffer(raw, dtype=_np.dtype(d["dtype"])).copy().reshape(d["shape"])
        if d.get("original_dtype") == "torch.bfloat16":
            arr = arr.view(_np.uint16).astype(_np.uint32).__lshift__(16).view(_np.float32)
        return arr


def deserialize_routing(raw: dict) -> dict | None:
    blob = raw.get("routing")
    if blob is None:
        return None
    try:
        return {
            "layer_indices":   blob.get("layer_indices", []),
            "topk_ids":        deserialize_tensor(blob["topk_ids"]),
            "topk_weights":    deserialize_tensor(blob["topk_weights"]),
            "routing_entropy": deserialize_tensor(blob["routing_entropy"]),
        }
    except Exception as e:
        print(f"[warn] routing deserialize: {e}")
        return None


def _extract_activations(raw, layers):
    blob = raw.get("activations")
    if blob is None:
        return None
    rs = blob.get("residual_stream")
    if rs is None:
        return None
    try:
        arr = np.asarray(deserialize_tensor(rs))
        if arr.ndim == 2:
            arr = arr[np.newaxis]
        return {layer_idx: arr[i] for i, layer_idx in enumerate(layers) if i < arr.shape[0]}
    except Exception as e:
        print(f"[warn] activations deserialize: {e}")
        return None


def build_pairs(n, seed=0):
    from datasets import load_dataset
    ds = load_dataset("lmsys/toxic-chat", "toxicchat0124", split="train")
    toxic = ds.filter(lambda r: r["toxicity"] == 1).shuffle(seed=seed).select(range(n))
    benign = ds.filter(lambda r: r["toxicity"] == 0).shuffle(seed=seed).select(range(n))
    return [(toxic[i]["user_input"], benign[i]["user_input"]) for i in range(n)]


def call(client, model, prompt, residual_layers, routing_layers, max_new_tokens, dump_path=None):
    t0 = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=max_new_tokens,
        extra_body={
            "vllm_xargs": {
                "output_residual_stream": residual_layers,
                "output_routing": routing_layers,
            },
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    elapsed = time.perf_counter() - t0
    raw = response.model_dump()
    if dump_path:
        dump_path.write_text(json.dumps(raw, indent=2, default=str))
        print(f"  [debug] raw response → {dump_path}")
    return {
        "text":       response.choices[0].message.content or "",
        "activations": _extract_activations(raw, residual_layers),
        "routing":     deserialize_routing(raw),
        "elapsed_s":   elapsed,
        "act_bytes":   len(json.dumps(raw.get("activations", {}), default=str).encode()),
        "rout_bytes":  len(json.dumps(raw.get("routing", {}), default=str).encode()),
    }


def check_routing(r, routing_layers, top_k=8):
    routing = r.get("routing")
    if routing is None:
        print("  [FAIL] routing key absent")
        return False
    ids = routing["topk_ids"]
    weights = routing["topk_weights"]
    entropy = routing["routing_entropy"]
    n = len(routing_layers)
    print(f"  topk_ids.shape      = {ids.shape}  (expect [{n}, n_tokens, {top_k}])")
    print(f"  topk_weights.shape  = {weights.shape}")
    print(f"  routing_entropy.shape = {entropy.shape}")
    ok = True
    if ids.shape[0] != n:
        print(f"  [FAIL] expected {n} layers, got {ids.shape[0]}")
        ok = False
    if ids.shape[-1] != top_k:
        print(f"  [FAIL] expected top_k={top_k}, got {ids.shape[-1]}")
        ok = False
    wmin, wmax = float(weights.min()), float(weights.max())
    if not (0.0 <= wmin and wmax <= 1.01):
        print(f"  [FAIL] weights out of [0,1]: min={wmin:.4f} max={wmax:.4f}")
        ok = False
    if float(entropy.min()) < 0:
        print(f"  [FAIL] negative entropy: {entropy.min():.4f}")
        ok = False
    if ok:
        print("  [OK] routing checks passed")
        print(f"       expert ids (layer 0, tok 0):  {ids[0, 0, :]}")
        print(f"       weights    (layer 0, tok 0):  {weights[0, 0, :]}")
        print(f"       entropy    (layer 0, tok 0):  {entropy[0, 0]:.4f} bits")
    return ok


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default=os.environ.get("VLLM_BASE_URL", "http://localhost:8001/v1"))
    p.add_argument("--api-key",  default=os.environ.get("VLLM_API_KEY", "EMPTY"))
    p.add_argument("--model",    default="glm-5-1")
    p.add_argument("--n-pairs",  type=int, default=3)
    p.add_argument("--residual-layers", type=int, nargs="+", default=[12, 39, 62])
    p.add_argument("--routing-layers",  type=int, nargs="+", default=[3, 39, 62])
    p.add_argument("--max-new-tokens",  type=int, default=32)
    p.add_argument("--out-dir",  type=Path, default=Path("./runs/phase2_smoke"))
    p.add_argument("--dump-first", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    from openai import OpenAI
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    print(f"[smoke] residual layers: {args.residual_layers}")
    print(f"[smoke] routing layers:  {args.routing_layers}")
    pairs = build_pairs(args.n_pairs)
    all_ok, timings, log = True, [], []
    for i, (toxic, benign) in enumerate(pairs):
        for label, prompt in [("toxic", toxic), ("benign", benign)]:
            print(f"\n--- pair {i} / {label} ---")
            dump = args.out_dir / f"raw_p0_{label}.json" if args.dump_first and i == 0 else None
            try:
                r = call(client, args.model, prompt, args.residual_layers,
                         args.routing_layers, args.max_new_tokens, dump_path=dump)
            except Exception as e:
                print(f"  [ERROR] {e}")
                all_ok = False
                continue
            ok = check_routing(r, args.routing_layers)
            all_ok = all_ok and ok
            timings.append(r["elapsed_s"])
            log.append({"pair": i, "label": label, "elapsed_s": r["elapsed_s"],
                        "act_bytes": r["act_bytes"], "rout_bytes": r["rout_bytes"]})
            print(f"  elapsed: {r['elapsed_s']:.2f}s  "
                  f"act: {r['act_bytes']//1024}KB  rout: {r['rout_bytes']//1024}KB")
    print(f"\n[smoke] mean {np.mean(timings):.2f}s  min {min(timings):.2f}s  max {max(timings):.2f}s")
    (args.out_dir / "timing.json").write_text(json.dumps(log, indent=2))
    if all_ok:
        print("\n[smoke] ALL CHECKS PASSED")
    else:
        print("\n[smoke] SOME CHECKS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
