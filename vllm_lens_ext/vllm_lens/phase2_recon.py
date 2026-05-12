"""
Phase 2 reconnaissance — static source inspection (no model load, no GPU work).

Purpose:
  EXTENSION_NOTES_PHASE2.md was written from public docs without full access
  to vllm-lens v1.1.0 source or vLLM's GLM 4 MoE source as installed in the
  Phase 1 container. Six "verify on actual hardware" items remain. This script
  knocks down four of them from already-installed source on the running debug
  container, with no model load and no interference with the live `vllm serve`
  process.

How to run:
  ssh -p <port> root@console.tinfoil.sh    # port from Tinfoil dashboard
  # then on the container, in a SECOND shell so the running serve is unaffected:
  cat > /tmp/phase2_recon.py <<'EOF'
  ... paste this file ...
  EOF
  python3 /tmp/phase2_recon.py 2>&1 | tee /tmp/phase2_recon.log
  # then `cat /tmp/phase2_recon.log` and paste back here.

Output is plaintext, ~10-30 KB. Read-only and idempotent.

Sections, and which EXTENSION_NOTES_PHASE2.md ⚠️ items each addresses:

  §1  vllm-lens installed source dump
        Verifies real package layout (notes say `src/vllm_lens/...`; reality
        per pyproject.toml is `vllm_lens/...`).
        Dumps _worker_ext.py, _activations_plugin.py, _helpers/_serialize.py,
        _helpers/_steering.py.
        → resolves §2 (collective_rpc method name) and §4 (setup_capture vs
        hook-guard split).

  §2  HiddenStatesExtension RPC surface (just `inspect`).
        → resolves §2 (lists exact RPC method names exposed).

  §3  vLLM FusedMoE + Glm4MoE source via inspect.getsource.
        → resolves §7 (FusedMoE pre-hook positional arg order).

  §4  CUDA graph + FusedMoE signals; greps vllm-lens for enforce_eager etc.
        → resolves §8/§9 (how vllm-lens handles CUDA graphs today).

  §5  GLM 5.1 DSA attention class source.
        → spike for §10 (post-decompress Q,K hook target for E2).
"""

from __future__ import annotations

import importlib
import importlib.metadata
import inspect
import sys
from pathlib import Path

SEP = "=" * 78


def banner(s: str) -> None:
    print(f"\n{SEP}\n{s}\n{SEP}", flush=True)


def safe(callable_, *args, **kwargs):
    try:
        return callable_(*args, **kwargs)
    except Exception as e:
        import traceback as tb
        print(f"[FAIL] {callable_.__name__}: {type(e).__name__}: {e}")
        tb.print_exc(limit=3)
        return None


def dump_path(p: Path, label: str | None = None) -> None:
    label = label or str(p)
    if not p.exists():
        print(f"\n[skip] {label} not present")
        return
    print(f"\n----- BEGIN {label} ({p.stat().st_size} B) -----")
    print(p.read_text())
    print(f"----- END {label} -----")


# §1 ------------------------------------------------------------------------

def dump_vllm_lens() -> Path | None:
    banner("§1  vllm-lens installed source")
    try:
        import vllm_lens  # type: ignore
    except Exception as e:
        print(f"[FAIL] import vllm_lens: {e}")
        return None

    pkg = Path(vllm_lens.__file__).parent
    print(f"package dir: {pkg}")
    try:
        ver = importlib.metadata.version("vllm-lens")
    except Exception:
        ver = "<unknown>"
    print(f"version:     {ver}")

    print("\n-- python files --")
    for f in sorted(pkg.rglob("*.py")):
        print(f"  {f.relative_to(pkg)}  ({f.stat().st_size} B)")

    for rel in [
        "__init__.py",
        "_activations_plugin.py",
        "_worker_ext.py",
        "_inspect_entry.py",
        "_helpers/_serialize.py",
        "_helpers/_steering.py",
    ]:
        dump_path(pkg / rel, label=f"vllm_lens/{rel}")

    return pkg


# §2 ------------------------------------------------------------------------

def dump_rpc_surface() -> None:
    banner("§2  HiddenStatesExtension RPC surface (inspect only)")
    try:
        from vllm_lens import _worker_ext  # type: ignore
    except Exception as e:
        print(f"[FAIL] import _worker_ext: {e}")
        return
    classes = [
        (n, c) for n, c in inspect.getmembers(_worker_ext, inspect.isclass)
        if c.__module__ == _worker_ext.__name__
    ]
    if not classes:
        print("[FAIL] no classes defined in _worker_ext")
        return
    for name, cls in classes:
        print(f"\nclass {name}:")
        for mname, fn in inspect.getmembers(cls, predicate=inspect.isfunction):
            if mname.startswith("__"):
                continue
            try:
                sig = str(inspect.signature(fn))
            except Exception:
                sig = "(?)"
            visibility = "private" if mname.startswith("_") else "PUBLIC (rpc-callable)"
            print(f"  [{visibility}] {mname}{sig}")


# §3 ------------------------------------------------------------------------

def dump_fused_moe() -> None:
    banner("§3  vLLM FusedMoE + Glm4MoE source (resolves pre-hook arg order)")
    # vLLM has shifted module paths several times; try a list of likely ones.
    candidates = [
        "vllm.model_executor.layers.fused_moe.layer",
        "vllm.model_executor.layers.fused_moe.fused_moe",
        "vllm.model_executor.layers.fused_moe",
        "vllm.model_executor.models.glm4_moe",
        "vllm.model_executor.models.glm5_moe",
        "vllm.model_executor.models.glm",
    ]
    for modname in candidates:
        try:
            m = importlib.import_module(modname)
        except Exception as e:
            print(f"\n[skip] {modname}: {type(e).__name__}: {e}")
            continue
        f = getattr(m, "__file__", None)
        print(f"\n-- module: {modname}  ({f}) --")
        for name, obj in inspect.getmembers(m, inspect.isclass):
            if obj.__module__ != modname:
                continue
            print(f"\n  class {name}({', '.join(b.__name__ for b in obj.__bases__)})")
            forward = getattr(obj, "forward", None)
            if forward is None:
                continue
            try:
                src = inspect.getsource(forward)
            except OSError:
                src = "    <source not available>"
            print(f"  --- {name}.forward ---")
            print("    " + src.replace("\n", "\n    "))


# §4 ------------------------------------------------------------------------

def dump_cudagraph_signals() -> None:
    banner("§4  CUDA graph signals in vllm-lens")
    needles = [
        "enforce_eager", "cuda_graph", "cudagraph", "compile",
        "register_forward_pre_hook", "register_forward_hook",
    ]
    for modname in ("vllm_lens._activations_plugin", "vllm_lens._worker_ext"):
        try:
            m = importlib.import_module(modname)
        except Exception as e:
            print(f"[FAIL] import {modname}: {e}")
            continue
        f = Path(m.__file__)
        print(f"\n-- {modname} ({f.name}) --")
        src = f.read_text()
        for n in needles:
            cnt = src.count(n)
            if cnt:
                print(f"  '{n}' x{cnt}:")
                for i, line in enumerate(src.splitlines(), start=1):
                    if n in line:
                        print(f"    L{i}: {line.strip()}")


# §5 ------------------------------------------------------------------------

def dump_dsa() -> None:
    banner("§5  GLM 5.1 DSA attention class (spike for E2 hook target)")
    for modname in (
        "vllm.model_executor.models.glm4_moe",
        "vllm.model_executor.models.glm5_moe",
        "vllm.model_executor.models.glm",
    ):
        try:
            m = importlib.import_module(modname)
        except Exception as e:
            print(f"\n[skip] {modname}: {type(e).__name__}: {e}")
            continue
        for name, obj in inspect.getmembers(m, inspect.isclass):
            if obj.__module__ != modname:
                continue
            if "Attn" not in name and "Attention" not in name:
                continue
            print(f"\n-- {modname}.{name} --")
            try:
                print(inspect.getsource(obj))
            except OSError:
                print("  <source not available>")


# main ----------------------------------------------------------------------

def main() -> int:
    banner("phase2 recon (static source only; safe to run during live serve)")
    print(f"python: {sys.version.split()[0]}")
    safe(dump_vllm_lens)
    safe(dump_rpc_surface)
    safe(dump_fused_moe)
    safe(dump_cudagraph_signals)
    safe(dump_dsa)
    banner("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())