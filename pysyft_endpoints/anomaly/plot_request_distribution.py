"""
plot_request_distribution.py — anomaly trigger for plot-request distribution.

Premise: a benign auditor's plot requests should be distributed across
endpoints in a way that's stable across sessions. A steganographic adversary
(encoding bits in the choice of which plot to request next) will produce a
distribution that deviates from the benign baseline.

Implementation:
  1. `BaselineCalibrator` consumes a stream of (endpoint_id,) tuples from
     prior benign sessions and produces a Dirichlet-smoothed probability
     vector over endpoint_ids.
  2. `Detector` maintains a sliding window of recent plot-bearing requests
     (the bundle.n_plots > 0 ones, per the engagement ledger). On each
     request it computes the symmetric KL divergence between the window's
     empirical distribution and the baseline, and fires when it exceeds
     `threshold_nats` (calibrated to a small fixed false-positive rate
     against the same baseline).

The detector is engagement-scoped: each engagement gets its own Detector
instance. The instance state (window + counts) lives in-process; on
endpoint-level restart, the detector rehydrates from the engagement ledger
by replaying the bundle table for the engagement.

Calibration for Phase 1: thresholds are set by the M4 experiment
(steganographic test corpus) so this file ships with `threshold_nats=None`
and the caller sets it after calibration. `threshold_nats=None` puts the
detector in report-only mode — KL is computed and logged but the trigger
never fires.
"""
from __future__ import annotations

import math
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Iterable, Optional


# Default endpoint vocabulary for the four-endpoint Phase 1 deployment.
# Extend if more endpoints land; the calibrator is robust to unseen
# endpoint_ids via the Dirichlet prior.
DEFAULT_ENDPOINT_VOCAB = (
    "prepilot.capture_residual_stream",
    "prepilot.capture_routing",
    "prepilot.capture_attention_stats",
    "prepilot.apply_steering",
)


@dataclass
class BaselineCalibrator:
    """Build a smoothed probability vector from benign request samples.

    Dirichlet prior with concentration `alpha_prior` per endpoint (default
    1.0 = uniform). Avoids zero-probability cells that would blow up the
    KL divergence on the first unseen endpoint.
    """
    vocab: tuple = DEFAULT_ENDPOINT_VOCAB
    alpha_prior: float = 1.0
    counts: Counter = field(default_factory=Counter)

    def observe(self, endpoint_id: str) -> None:
        self.counts[endpoint_id] += 1

    def observe_many(self, endpoint_ids: Iterable[str]) -> None:
        for eid in endpoint_ids:
            self.counts[eid] += 1

    def probabilities(self) -> dict:
        total = sum(self.counts.get(v, 0) for v in self.vocab) \
                + self.alpha_prior * len(self.vocab)
        return {
            v: (self.counts.get(v, 0) + self.alpha_prior) / total
            for v in self.vocab
        }

    def n_observations(self) -> int:
        return sum(self.counts.get(v, 0) for v in self.vocab)


def _symmetric_kl(p: dict, q: dict) -> float:
    """Symmetric KL = 0.5 * (KL(p||q) + KL(q||p)). Both p and q must be
    smoothed (no zero entries) — the BaselineCalibrator's Dirichlet prior
    guarantees this for q; the Detector smooths p the same way."""
    keys = set(p.keys()) | set(q.keys())
    kl_pq = sum(p.get(k, 0.0) * math.log(p.get(k, 0.0) / q.get(k, 1e-12))
                 for k in keys if p.get(k, 0.0) > 0)
    kl_qp = sum(q.get(k, 0.0) * math.log(q.get(k, 0.0) / p.get(k, 1e-12))
                 for k in keys if q.get(k, 0.0) > 0)
    return 0.5 * (kl_pq + kl_qp)


@dataclass
class Detector:
    """Sliding-window symmetric-KL anomaly detector.

    Window size, prior, and threshold are calibration knobs. M4 (handoff
    §3.4 run 3) calibrates them on a benign + steganographic test corpus.
    Until M4 runs, `threshold_nats` is None and the detector reports KL
    on every fire-check but never raises the trigger.
    """
    baseline: BaselineCalibrator
    window_size: int = 50
    alpha_prior: float = 1.0
    threshold_nats: Optional[float] = None

    _window: deque = field(default_factory=lambda: deque(maxlen=50))

    def __post_init__(self) -> None:
        # Reconfigure the deque's maxlen to match the configured window_size
        # (dataclass default_factory ran before window_size was bound).
        self._window = deque(self._window, maxlen=self.window_size)

    def observe(self, endpoint_id: str) -> dict:
        """Record a request and return current status.

        Returns:
            {
              "kl_nats": float,            # current symmetric KL vs baseline
              "window_size": int,          # actual current window size
              "fired": bool,               # True iff threshold crossed
              "threshold_nats": float|None # the configured threshold
            }
        """
        self._window.append(endpoint_id)
        # Build smoothed empirical distribution from the window.
        counts = Counter(self._window)
        total = len(self._window) + self.alpha_prior * len(self.baseline.vocab)
        empirical = {
            v: (counts.get(v, 0) + self.alpha_prior) / total
            for v in self.baseline.vocab
        }
        kl = _symmetric_kl(empirical, self.baseline.probabilities())
        fired = (
            self.threshold_nats is not None
            and len(self._window) >= self.window_size
            and kl >= self.threshold_nats
        )
        return {
            "kl_nats": float(kl),
            "window_size": len(self._window),
            "fired": fired,
            "threshold_nats": self.threshold_nats,
        }

    def rehydrate_from_history(self, history: Iterable[str]) -> None:
        """Replay endpoint_ids from the engagement ledger to repopulate the
        window after a restart. Caller passes the last `window_size`
        endpoint_ids in chronological order."""
        self._window.clear()
        for eid in history:
            self._window.append(eid)
