# cc-deep-eval

Tinfoil-deployed vLLM target images for [*An empirical study of Confidential Compute for frontier AI evaluations*](https://pourdemain.ngo/) (Pour Demain, May 2026). This repo contains the Dockerfiles, vllm-lens plugin, PySyft Datasite server, egress encoder, and server-side endpoint definitions that run inside Intel TDX enclaves on 8×H200 SXM.

The companion repo [`cc-benchbox`](https://github.com/pourdemain/cc-benchbox) contains the laptop-side measurement harness (drivers, analysis, prompt data).

## What this repo does

Builds Docker images deployed as Tinfoil Containers. Each image wraps vLLM 0.20.0 with the vllm-lens instrumentation plugin and (for governed-egress images) a PySyft 0.9.5 Datasite with identity-gated capture endpoints and an engagement-budget ledger.

## Structure

```
images
  Dockerfile                     # GLM-5.1-FP8 primary (vLLM + vllm-lens)
  Dockerfile.gradient            # Gradient sidecar (transformers + HF FineGrainedFP8)
  Dockerfile.egress              # Tier-1 egress pipeline (vLLM + egress encoder)
  Dockerfile.pysyft              # PySyft governed-egress (vLLM + Datasite + endpoints)

vllm-lens plugin
  vllm_lens_ext/                 # installable plugin package
    vllm_lens/_activations_plugin.py   # residual/routing/attention capture + steering
    vllm_lens/_routing_ext.py          # MoE routing hook
    vllm_lens/_gradient_backend.py     # gradient sidecar backend
    vllm_lens/_gradient_entry.py       # gradient HTTP endpoint
    vllm_lens/_helpers/_serialize.py   # tensor serialization (zstd + base64)
    vllm_lens/_inspect_entry.py        # Inspect-AI provider entry

server-side services
  egress_service.py              # Tier-1 encoder (aggregate, plot, sign, ledger)
  gradient_server.py             # Gradient sidecar HTTP server
  pysyft_datasite_server.py      # PySyft Datasite launcher + endpoint registration
  register_endpoints.py          # hot-swap endpoint registration (admin)
  register_auditors.py           # auditor user creation
  ledger_admin.py                # runtime budget control

PySyft endpoints
  pysyft_endpoints/
    endpoints/_common.py         # shared: identity gate, loopback call, timing
    endpoints/capture_residual_stream.py
    endpoints/capture_routing.py
    endpoints/capture_attention_stats.py
    endpoints/apply_steering.py
    ledger/engagement_ledger.py  # SQLite engagement/session/bundle ledger
    ledger/schema.sql            # ledger DDL
    anomaly/plot_request_distribution.py  # request-channel monitor

entrypoints
  entrypoint_pysyft.sh           # stale-shm cleanup + supervisor (vLLM + egress + Datasite)
  entrypoint_egress.sh           # vLLM + egress encoder

deployment configs
  tinfoil-config.yml             # GLM-5.1-FP8 primary
  tinfoil-config-egress.yml      # Tier-1 egress
  tinfoil-config-pysyft.yml      # PySyft governed-egress

Phase 1/2 validation scripts (historical)
  scripts/
    captures.py                  # client-side capture library
    phase2_capture.py            # Phase 2 4-condition validation driver
    d6_steering.py               # RepE steering smoke test
    build_pairs.py               # ToxicChat pair builder
```

## Image tags

| Image | Tag | Brief section |
|-------|-----|---------------|
| `prepilot-vllm-lens` | `v0.0.25` | §2–§3 (primary matrix) |
| `prepilot-vllm-lens-grad` | `v0.1.16-grad` | §2 (gradient sidecar) |
| `prepilot-vllm-lens` | `v0.0.27-llama70b-1` | §3.3 (Llama-70B TP=8) |
| `prepilot-vllm-lens` | `v0.0.28-llama70b-tp1` | §3.3 (Llama-70B TP=1) |
| `prepilot-vllm-lens` | `v0.0.32-egress` | §4 (egress pipeline) |
| `prepilot-vllm-lens` | `v0.0.9-pysyft` | §5 (governed-egress) |

Each tag has a corresponding Sigstore attestation via the Tinfoil build workflow.

## License

Creative Commons Attribution 4.0 International (CC-BY-4.0)
