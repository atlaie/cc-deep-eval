"""
Architecture:
  - This script runs on the user's laptop, NOT inside the Tinfoil enclave.
  - Hits the public Tinfoil container URL (https://<name>.<org>.containers.tinfoil.dev)
    via /v1/chat/completions, with extra_args.output_residual_stream.
  - Auth: Authorization: Bearer <tinfoil-api-key> (set via VLLM_API_KEY env var).
  - Deserializes activations using vllm_lens._helpers._serialize.deserialize_tensor.
    If vllm-lens isn't installable on the laptop (CUDA-bound transitive deps),
    the script gracefully degrades to text-only generation logging and prints
    a warning; we then vendor the deserializer at that point.
  - Outputs to ./runs/phase1_sanity/.

GLM 5.1 architecture facts (from zai-org/GLM-5.1-FP8 config.json):
  num_hidden_layers : 78 (3 dense + 75 MoE, first_k_dense_replace=3)
  hidden_size       : 6144
  num_attention_heads: 64
  n_routed_experts  : 256, top-8 per token
  n_shared_experts  : 1
  kv_lora_rank      : 512 (low-rank KV compression)
  q_lora_rank       : 2048
  max_position_embeddings: 202752
  vocab_size        : 154880
  dtype             : bfloat16 (deployed fp8 by Tinfoil)

Probe layers: [12, 23, 39, 51, 62, 70]
  All fall in the MoE band (layers 3-77). Layer fractions:
  0.15, 0.30, 0.50, 0.65, 0.80, 0.90 of 78 total layers.
  Mid-to-late band (39-62) is where RepE-style work finds the cleanest
  concept signal. Layer 12 and 70 are early/late diagnostic anchors.

Memory budget for activation capture (all 100 prompts, 6 layers, avg 100 tokens):
  6144 (hidden) * 4 (fp32) * 6 (layers) * 100 (tokens) * 100 (prompts)
  = ~1.4 GB. Well within 2.5 TB ramdisk.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_dataset
from openai import OpenAI

try:
    from vllm_lens._helpers._serialize import deserialize_tensor  # type: ignore
except Exception:
    deserialize_tensor = None


# --------- GLM 5.1 architecture constants ------------------------------------

GLM51_NUM_LAYERS = 78
GLM51_HIDDEN_SIZE = 6144
GLM51_FIRST_DENSE_LAYERS = 3  # layers 0-2 are dense; 3-77 are MoE

# Probe layers: fractional positions [0.15, 0.30, 0.50, 0.65, 0.80, 0.90]
# applied to 78 layers. All fall in the MoE band.
DEFAULT_LAYERS = [12, 23, 39, 51, 62, 70]


# --------- defaults ----------------------------------------------------------

DEFAULT_BASE_URL = "http://localhost:8001/v1"
DEFAULT_MODEL_NAME = "glm-5-1"  # matches Tinfoil's --served-model-name
DEFAULT_N_PAIRS = 50
DEFAULT_MAX_PROMPT_CHARS = 1024
DEFAULT_OUT_DIR = Path("./runs/phase1_sanity")


# --------- args --------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default=os.environ.get("VLLM_BASE_URL", DEFAULT_BASE_URL))
    p.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", "EMPTY"))
    p.add_argument("--model", default=DEFAULT_MODEL_NAME)
    p.add_argument("--n-pairs", type=int, default=DEFAULT_N_PAIRS)
    p.add_argument("--layers", type=int, nargs="+", default=DEFAULT_LAYERS,
                   help="Layer indices to probe. Default probes are calibrated "
                        "for GLM 5.1's 78-layer architecture. If you change "
                        "model, recalculate.")
    p.add_argument("--max-prompt-chars", type=int, default=DEFAULT_MAX_PROMPT_CHARS)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dump-first-response", action="store_true",
                   help="Dump the raw HTTP response of the first call. Use on "
                        "the very first run to verify the activation schema.")
    return p.parse_args()


# --------- data --------------------------------------------------------------

@dataclass
class Pair:
    pair_id: int
    toxic: str
    benign: str


def build_contrastive_pairs(n_pairs: int, max_prompt_chars: int, seed: int) -> list[Pair]:
    """Pull toxicchat0124 and assemble n_pairs of toxic/benign user prompts."""
    ds = load_dataset("lmsys/toxic-chat", "toxicchat0124", split="train")
    toxic = ds.filter(lambda r: r["toxicity"] == 1 and len(r["user_input"]) < max_prompt_chars)
    benign = ds.filter(lambda r: r["toxicity"] == 0 and len(r["user_input"]) < max_prompt_chars)

    if len(toxic) < n_pairs or len(benign) < n_pairs:
        raise RuntimeError(
            f"Not enough samples after filter: toxic={len(toxic)} benign={len(benign)} "
            f"requested={n_pairs}. Loosen --max-prompt-chars or reduce --n-pairs."
        )

    toxic = toxic.shuffle(seed=seed).select(range(n_pairs))
    benign = benign.shuffle(seed=seed).select(range(n_pairs))

    return [
        Pair(pair_id=i, toxic=toxic[i]["user_input"], benign=benign[i]["user_input"])
        for i in range(n_pairs)
    ]


# --------- API client + activation extraction --------------------------------

def call_with_capture(
    client: OpenAI,
    model: str,
    user_prompt: str,
    layers: list[int],
    max_new_tokens: int,
    dump_path: Path | None = None,
) -> dict[str, Any]:
    """One chat-completion call with residual-stream capture.

    Returns: {"text": str, "activations": dict[int, np.ndarray] | None}
    """
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": user_prompt}],
        temperature=0.0,
        max_tokens=max_new_tokens,
        extra_body={
        "extra_args": {"output_residual_stream": layers},
        "chat_template_kwargs": {"enable_thinking": False},
    },
    )
    raw = response.model_dump()

    if dump_path is not None:
        dump_path.write_text(json.dumps(raw, indent=2, default=str))
        print(f"[debug] raw response dumped to {dump_path}")

    text = response.choices[0].message.content or ""
    activations = _extract_activations(raw)
    return {"text": text, "activations": activations}


def _extract_activations(raw: dict) -> dict[int, np.ndarray] | None:
    """Find residual_stream activations in the HTTP response.

    vLLM-Lens injects activations into the response before it's sent to
    the client. The exact key path is not publicly documented; the script
    tries four candidate locations. If all fail, re-run with
    --dump-first-response, find the key, and update the candidates list.
    """
    if deserialize_tensor is None:
        print("[warn] deserialize_tensor not importable; activations skipped. "
              "Check vllm-lens install.")
        return None

    candidates = [
        ("metadata", "activations", "residual_stream"),
        ("activations", "residual_stream"),
        ("choices", 0, "metadata", "activations", "residual_stream"),
        ("choices", 0, "activations", "residual_stream"),
    ]
    for path in candidates:
        node = raw
        ok = True
        for k in path:
            try:
                node = node[k]
            except (KeyError, IndexError, TypeError):
                ok = False
                break
        if ok and node is not None:
            return _deserialize_layer_dict(node)

    return None


def _deserialize_layer_dict(payload) -> dict[int, np.ndarray]:
    """Convert {layer_idx: serialized_tensor} → {int: np.ndarray}."""
    out: dict[int, np.ndarray] = {}
    if not isinstance(payload, dict):
        return {}
    for k, v in payload.items():
        try:
            layer = int(k)
        except (ValueError, TypeError):
            continue
        try:
            t = deserialize_tensor(v)  # type: ignore[misc]
            out[layer] = t.float().cpu().numpy() if hasattr(t, "cpu") else np.asarray(t)
        except Exception as e:
            print(f"[warn] deserialize layer {layer} failed: {e}")
    return out


# --------- diagnostics -------------------------------------------------------

def quick_diagnostics(
    toxic_results: list[dict],
    benign_results: list[dict],
    layers: list[int],
) -> dict[int, dict[str, float]]:
    """Per-layer difference-in-means diagnostic on last-token residual stream.

    For GLM 5.1 (hidden=6144), expected pattern if capture is working:
      - Layer 12 (early): cos_similarity ~0.97-0.99, diff_norm low
      - Layer 39 (mid):   cos_similarity ~0.90-0.95, diff_norm rising
      - Layer 62 (late):  cos_similarity <0.88, diff_norm near peak
    Flat across all layers = capture broken or data not contrastive.
    """
    summary: dict[int, dict[str, float]] = {}
    for L in layers:
        toxic_vecs = _last_token_vecs(toxic_results, L)
        benign_vecs = _last_token_vecs(benign_results, L)
        if not toxic_vecs or not benign_vecs:
            continue
        tm = np.stack(toxic_vecs).mean(axis=0)
        bm = np.stack(benign_vecs).mean(axis=0)
        cos = float(np.dot(tm, bm) / (np.linalg.norm(tm) * np.linalg.norm(bm) + 1e-8))
        summary[L] = {
            "n_toxic": len(toxic_vecs),
            "n_benign": len(benign_vecs),
            "diff_norm": float(np.linalg.norm(tm - bm)),
            "toxic_norm": float(np.linalg.norm(tm)),
            "benign_norm": float(np.linalg.norm(bm)),
            "cos_similarity": cos,
        }
    return summary


def _last_token_vecs(results: list[dict], layer: int) -> list[np.ndarray]:
    out = []
    for r in results:
        acts = r.get("activations") or {}
        if layer not in acts:
            continue
        t = acts[layer]
        # Expected shape: [seq_len, 6144] or [1, seq_len, 6144]
        if t.ndim == 3:
            t = t[0]
        out.append(t[-1])
    return out


# --------- main --------------------------------------------------------------

def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[phase1] model={args.model} base_url={args.base_url}")
    print(f"[phase1] probing layers: {args.layers}")
    print(f"[phase1] GLM 5.1: {GLM51_NUM_LAYERS} total layers, "
          f"{GLM51_FIRST_DENSE_LAYERS} dense + "
          f"{GLM51_NUM_LAYERS - GLM51_FIRST_DENSE_LAYERS} MoE")

    print(f"[phase1] building contrastive set: {args.n_pairs} pairs")
    pairs = build_contrastive_pairs(args.n_pairs, args.max_prompt_chars, args.seed)
    (args.out_dir / "pairs.json").write_text(
        json.dumps([p.__dict__ for p in pairs], indent=2)
    )

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    def _do_pass(prompts: list[str], label: str) -> list[dict]:
        results = []
        for i, p in enumerate(prompts):
            dump = (args.out_dir / f"raw_first_{label}.json"
                    if args.dump_first_response and i == 0 else None)
            try:
                r = call_with_capture(
                    client, args.model, p, args.layers, args.max_new_tokens,
                    dump_path=dump,
                )
                results.append({"prompt": p, **r})
            except Exception as e:
                print(f"[error] pair {i} ({label}): {e}")
                results.append({"prompt": p, "text": "", "activations": None, "error": str(e)})
            if (i + 1) % 10 == 0:
                print(f"  [{label}] {i+1}/{len(prompts)}")
        return results

    print("[phase1] capture pass: toxic")
    toxic_results = _do_pass([p.toxic for p in pairs], "toxic")
    print("[phase1] capture pass: benign")
    benign_results = _do_pass([p.benign for p in pairs], "benign")

    _save_npz(args.out_dir / "toxic.npz", toxic_results, args.layers)
    _save_npz(args.out_dir / "benign.npz", benign_results, args.layers)
    (args.out_dir / "generations.json").write_text(json.dumps(
        {
            "toxic": [{"prompt": r["prompt"], "text": r["text"]} for r in toxic_results],
            "benign": [{"prompt": r["prompt"], "text": r["text"]} for r in benign_results],
        }, indent=2,
    ))

    print("[phase1] diagnostics")
    summary = quick_diagnostics(toxic_results, benign_results, args.layers)
    (args.out_dir / "diagnostics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    n_captured = sum(1 for r in toxic_results if r.get("activations"))
    print(f"\n[phase1] activations captured: {n_captured}/{len(toxic_results)} toxic prompts")

    if not summary:
        print("\n[!!] No activations extracted.")
        print("     Re-run with --dump-first-response and inspect")
        print("     out_dir/raw_first_toxic.json to find the correct schema.")
        print("     Then patch _extract_activations() candidates list.")


def _save_npz(path: Path, results: list[dict], layers: list[int]) -> None:
    """Stack [n_pairs, hidden_size] arrays per layer (last-token only)."""
    bundles: dict[str, np.ndarray] = {}
    for L in layers:
        vecs = _last_token_vecs(results, L)
        if not vecs:
            continue
        bundles[f"layer_{L:03d}_last_tok"] = np.stack(vecs)
    if bundles:
        np.savez(path, **bundles)


if __name__ == "__main__":
    main()
