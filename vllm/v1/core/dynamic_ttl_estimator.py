#!/usr/bin/env python3
"""
Dynamic TTL estimator for Continuum-style KV-cache pinning.

This file is intentionally independent from vLLM so that it can be:

1. tested with synthetic tool-call traces;
2. imported by the preview Continuum code later;
3. compared against the preview repository's current 0/2-second policy.

Paper-aligned core objective (Continuum Sec. 4.1/4.2):

    score(tau) =
        P(tool_duration <= tau)
        * (average_queue_delay * memoryfulness + prefill_reload_cost)
        - tau

The estimator enumerates unique historical tool durations plus tau=0 and
returns the candidate with the highest expected score.

Cold-start hierarchy:

    insufficient global history -> fixed default TTL
    sufficient global history but insufficient per-tool history -> global CDF
    sufficient per-tool history -> per-tool CDF

Notes:
- The paper uses K=100. This implementation keeps it configurable.
- The preview repository uses a fixed 2-second threshold. We retain 2 seconds
  as the default cold-start TTL for compatibility, rather than claiming that
  this constant is a complete reproduction of the paper's derivation.
- The memoryfulness tracker below is a practical estimator based on completed
  programs. It is isolated so it can be replaced without changing TTL search.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import statistics
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Iterable, Mapping, Optional, Protocol, Sequence


class PrefillReloadProfile(Protocol):
    """Interface for model-hardware-specific prefill/reload profiling."""

    def estimate_seconds(self, context_tokens: int) -> float:
        """Return estimated prefill or reload cost in seconds."""


@dataclass(frozen=True)
class ConstantPrefillReloadProfile:
    """Useful for unit tests and early synthetic experiments."""

    seconds: float = 0.0

    def estimate_seconds(self, context_tokens: int) -> float:
        del context_tokens
        return max(0.0, float(self.seconds))


@dataclass(frozen=True)
class QuadraticPrefillReloadProfile:
    """Quadratic profile: a*n^2 + b*n + c seconds for n context tokens."""

    a: float
    b: float
    c: float

    def estimate_seconds(self, context_tokens: int) -> float:
        n = max(0, int(context_tokens))
        return max(0.0, self.a * n * n + self.b * n + self.c)


@dataclass(frozen=True)
class PiecewiseLinearPrefillReloadProfile:
    """
    Interpolate measured (context_tokens, seconds) points.

    This is convenient before fitting the quadratic curve described in the
    paper. Values outside the measured range use endpoint extrapolation.
    """

    points: Sequence[tuple[int, float]]

    def __post_init__(self) -> None:
        cleaned = sorted((int(x), float(y)) for x, y in self.points)
        if not cleaned:
            raise ValueError("points must not be empty")
        if any(x < 0 or y < 0 for x, y in cleaned):
            raise ValueError("profile points must be non-negative")
        if len({x for x, _ in cleaned}) != len(cleaned):
            raise ValueError("context-token values must be unique")
        object.__setattr__(self, "points", tuple(cleaned))

    def estimate_seconds(self, context_tokens: int) -> float:
        n = max(0, int(context_tokens))
        pts = self.points

        if len(pts) == 1:
            return pts[0][1]

        if n <= pts[0][0]:
            left, right = pts[0], pts[1]
        elif n >= pts[-1][0]:
            left, right = pts[-2], pts[-1]
        else:
            left = pts[0]
            right = pts[-1]
            for i in range(1, len(pts)):
                if n <= pts[i][0]:
                    left, right = pts[i - 1], pts[i]
                    break

        x0, y0 = left
        x1, y1 = right
        if x1 == x0:
            return max(0.0, y0)

        ratio = (n - x0) / (x1 - x0)
        return max(0.0, y0 + ratio * (y1 - y0))


@dataclass(frozen=True)
class TTLEstimatorConfig:
    """Configuration for the dynamic estimator."""

    history_threshold: int = 100
    default_ttl_seconds: float = 2.0
    queue_delay_window: int = 100
    max_tool_history: int = 10_000
    max_global_history: int = 50_000
    memoryfulness_window: int = 10_000
    default_memoryfulness: float = 1.0
    max_candidate_ttl_seconds: Optional[float] = None

    def __post_init__(self) -> None:
        if self.history_threshold < 1:
            raise ValueError("history_threshold must be >= 1")
        if self.default_ttl_seconds < 0:
            raise ValueError("default_ttl_seconds must be >= 0")
        if self.queue_delay_window < 1:
            raise ValueError("queue_delay_window must be >= 1")
        if self.max_tool_history < 1 or self.max_global_history < 1:
            raise ValueError("history limits must be >= 1")
        if self.memoryfulness_window < 2:
            raise ValueError("memoryfulness_window must be >= 2")
        if not -1.0 <= self.default_memoryfulness <= 1.0:
            raise ValueError("default_memoryfulness must be within [-1, 1]")
        if (
            self.max_candidate_ttl_seconds is not None
            and self.max_candidate_ttl_seconds < 0
        ):
            raise ValueError("max_candidate_ttl_seconds must be >= 0")


@dataclass(frozen=True)
class TTLEstimationResult:
    """Detailed result for logging, debugging, and experiments."""

    ttl_seconds: float
    history_source: str
    expected_score: float
    finish_probability: float
    average_queue_delay: float
    memoryfulness: float
    prefill_reload_cost: float
    candidate_count: int
    selected_history_size: int
    global_history_size: int
    tool_history_size: int


class MemoryfulnessTracker:
    """
    Practical online estimator for eta = -Corr(k, N-k).

    When a program completes with N total requests, this tracker adds the
    intermediate samples:

        (k, N-k), k = 1, ..., N-1

    The paper defines eta at workload level but does not release the exact
    estimator implementation. Keeping this logic in a separate class makes
    the approximation explicit and replaceable.
    """

    def __init__(self, max_samples: int, default_eta: float = 1.0) -> None:
        self._k_values: Deque[float] = deque(maxlen=max_samples)
        self._remaining_values: Deque[float] = deque(maxlen=max_samples)
        self._default_eta = float(default_eta)

    def record_completed_program(self, total_requests: int) -> None:
        n = int(total_requests)
        if n < 2:
            return

        for k in range(1, n):
            self._k_values.append(float(k))
            self._remaining_values.append(float(n - k))

    def estimate(self) -> float:
        if len(self._k_values) < 2:
            return self._default_eta

        x = list(self._k_values)
        y = list(self._remaining_values)
        mean_x = statistics.fmean(x)
        mean_y = statistics.fmean(y)

        covariance = sum(
            (xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y)
        )
        variance_x = sum((xi - mean_x) ** 2 for xi in x)
        variance_y = sum((yi - mean_y) ** 2 for yi in y)

        if variance_x <= 0.0 or variance_y <= 0.0:
            return self._default_eta

        corr = covariance / math.sqrt(variance_x * variance_y)
        eta = -corr
        return min(1.0, max(-1.0, eta))


class DynamicTTLEstimator:
    """
    Standalone implementation of the missing Continuum TTL calculation.

    Runtime integration only needs to provide:
    - tool names and observed tool durations;
    - queueing delays for evicted requests;
    - context length, or a directly measured prefill/reload cost;
    - completed program lengths if online memoryfulness is desired.
    """

    def __init__(
        self,
        config: Optional[TTLEstimatorConfig] = None,
        prefill_reload_profile: Optional[PrefillReloadProfile] = None,
    ) -> None:
        self.config = config or TTLEstimatorConfig()
        self.prefill_reload_profile = (
            prefill_reload_profile or ConstantPrefillReloadProfile(0.0)
        )
        self.cdf_implementation = os.environ.get(
            "CONTINUUM_TTL_CDF_IMPL", "optimized"
        ).strip().lower()
        if self.cdf_implementation not in {"naive", "optimized"}:
            raise ValueError(
                "CONTINUUM_TTL_CDF_IMPL must be naive or optimized, got "
                f"{self.cdf_implementation!r}"
            )

        self._tool_histories: dict[str, Deque[float]] = defaultdict(
            lambda: deque(maxlen=self.config.max_tool_history)
        )
        self._global_history: Deque[float] = deque(
            maxlen=self.config.max_global_history
        )
        self._queue_delays: Deque[float] = deque(
            maxlen=self.config.queue_delay_window
        )
        self._memoryfulness = MemoryfulnessTracker(
            max_samples=self.config.memoryfulness_window,
            default_eta=self.config.default_memoryfulness,
        )

    # ------------------------------------------------------------------
    # Runtime observations
    # ------------------------------------------------------------------

    def record_tool_duration(self, tool: str, duration_seconds: float) -> None:
        tool_name = str(tool).strip()
        duration = float(duration_seconds)

        if not tool_name:
            raise ValueError("tool must not be empty")
        if not math.isfinite(duration) or duration < 0:
            raise ValueError("duration_seconds must be finite and >= 0")

        self._tool_histories[tool_name].append(duration)
        self._global_history.append(duration)

    def record_queue_delay(self, delay_seconds: float) -> None:
        delay = float(delay_seconds)
        if not math.isfinite(delay) or delay < 0:
            raise ValueError("delay_seconds must be finite and >= 0")
        self._queue_delays.append(delay)

    def record_completed_program(self, total_requests: int) -> None:
        self._memoryfulness.record_completed_program(total_requests)

    # ------------------------------------------------------------------
    # Read-only state
    # ------------------------------------------------------------------

    def get_tool_history(self, tool: str) -> tuple[float, ...]:
        return tuple(self._tool_histories.get(tool, ()))

    def get_global_history(self) -> tuple[float, ...]:
        return tuple(self._global_history)

    def average_queue_delay(self) -> float:
        if not self._queue_delays:
            return 0.0
        return statistics.fmean(self._queue_delays)

    def memoryfulness(self) -> float:
        return self._memoryfulness.estimate()

    # ------------------------------------------------------------------
    # Paper-aligned TTL calculation
    # ------------------------------------------------------------------

    @staticmethod
    def empirical_finish_probability(
        history: Sequence[float],
        ttl_seconds: float,
    ) -> float:
        if not history:
            return 0.0
        ttl = float(ttl_seconds)
        return sum(duration <= ttl for duration in history) / len(history)

    def _select_history(self, tool: str) -> tuple[str, Optional[tuple[float, ...]]]:
        global_history = tuple(self._global_history)
        tool_history = tuple(self._tool_histories.get(tool, ()))
        k = self.config.history_threshold

        # Paper: use T_default when |S| <= K.
        if len(global_history) <= k:
            return "fixed_cold_start", None

        # Paper: use global records when |S[f]| <= K.
        if len(tool_history) <= k:
            return "global_history", global_history

        return "per_tool_history", tool_history

    def _candidate_ttls(self, history: Sequence[float]) -> list[float]:
        candidates = {0.0}

        for duration in history:
            duration = float(duration)
            if duration < 0 or not math.isfinite(duration):
                continue
            if (
                self.config.max_candidate_ttl_seconds is not None
                and duration > self.config.max_candidate_ttl_seconds
            ):
                continue
            candidates.add(duration)

        return sorted(candidates)

    def estimate_ttl(
        self,
        tool: Optional[str],
        context_tokens: int = 0,
        *,
        queue_delay_seconds: Optional[float] = None,
        memoryfulness: Optional[float] = None,
        prefill_reload_cost_seconds: Optional[float] = None,
    ) -> TTLEstimationResult:
        """
        Estimate TTL for one finished request.

        Optional keyword arguments override online/profiled values. This keeps
        the estimator independent from vLLM and simplifies controlled tests.
        """

        tool_name = "" if tool is None else str(tool).strip()
        global_size = len(self._global_history)
        tool_size = len(self._tool_histories.get(tool_name, ()))

        if not tool_name:
            return TTLEstimationResult(
                ttl_seconds=0.0,
                history_source="no_tool_call",
                expected_score=0.0,
                finish_probability=0.0,
                average_queue_delay=0.0,
                memoryfulness=self.config.default_memoryfulness,
                prefill_reload_cost=0.0,
                candidate_count=1,
                selected_history_size=0,
                global_history_size=global_size,
                tool_history_size=tool_size,
            )

        queue_delay = (
            self.average_queue_delay()
            if queue_delay_seconds is None
            else max(0.0, float(queue_delay_seconds))
        )
        eta = (
            self.memoryfulness()
            if memoryfulness is None
            else min(1.0, max(-1.0, float(memoryfulness)))
        )
        prefill_reload = (
            self.prefill_reload_profile.estimate_seconds(context_tokens)
            if prefill_reload_cost_seconds is None
            else max(0.0, float(prefill_reload_cost_seconds))
        )

        source, history = self._select_history(tool_name)

        if history is None:
            ttl = self.config.default_ttl_seconds
            return TTLEstimationResult(
                ttl_seconds=ttl,
                history_source=source,
                expected_score=0.0,
                finish_probability=0.0,
                average_queue_delay=queue_delay,
                memoryfulness=eta,
                prefill_reload_cost=prefill_reload,
                candidate_count=1,
                selected_history_size=0,
                global_history_size=global_size,
                tool_history_size=tool_size,
            )

        candidates = self._candidate_ttls(history)
        benefit_if_hit = queue_delay * eta + prefill_reload

        best_ttl = 0.0
        best_score = 0.0
        best_probability = self.empirical_finish_probability(history, 0.0)

        # CONTINUUM_TTL_CDF_OPT_V1
        if self.cdf_implementation == "optimized":
            # Mathematically equivalent empirical-CDF search with one sorted
            # history scan: O(n log n) instead of O(n^2).
            sorted_history = sorted(float(duration) for duration in history)
            history_size = len(sorted_history)
            history_index = 0

            for ttl in candidates:
                while history_index < history_size and sorted_history[history_index] <= ttl:
                    history_index += 1
                probability = history_index / history_size
                score = probability * benefit_if_hit - ttl
                if score > best_score + 1e-12 or (abs(score - best_score) <= 1e-12 and ttl < best_ttl):
                    best_ttl = ttl
                    best_score = score
                    best_probability = probability
        else:
            # Historical paper-aligned implementation retained for controlled
            # ablation and reproducibility.
            for ttl in candidates:
                probability = self.empirical_finish_probability(history, ttl)
                score = probability * benefit_if_hit - ttl
                if score > best_score + 1e-12 or (abs(score - best_score) <= 1e-12 and ttl < best_ttl):
                    best_ttl = ttl
                    best_score = score
                    best_probability = probability

        return TTLEstimationResult(
            ttl_seconds=best_ttl,
            history_source=source,
            expected_score=best_score,
            finish_probability=best_probability,
            average_queue_delay=queue_delay,
            memoryfulness=eta,
            prefill_reload_cost=prefill_reload,
            candidate_count=len(candidates),
            selected_history_size=len(history),
            global_history_size=global_size,
            tool_history_size=tool_size,
        )


# ----------------------------------------------------------------------
# Preview-code baseline
# ----------------------------------------------------------------------

def preview_repository_ttl(
    tool: Optional[str],
    per_tool_history: Mapping[str, Sequence[float]],
    fixed_threshold_seconds: float = 2.0,
) -> float:
    """
    Reproduce the preview repository's current set_up_pin() behavior.

    No tool call -> 0
    Historical mean > 2 seconds -> 0
    Otherwise -> 2 seconds

    A new tool has no history, so its mean is treated as zero.
    """

    tool_name = "" if tool is None else str(tool).strip()
    if not tool_name:
        return 0.0

    history = per_tool_history.get(tool_name, ())
    mean_duration = statistics.fmean(history) if history else 0.0

    if mean_duration > fixed_threshold_seconds:
        return 0.0
    return float(fixed_threshold_seconds)


# ----------------------------------------------------------------------
# Synthetic comparison
# ----------------------------------------------------------------------

@dataclass
class PolicyMetrics:
    requests: int = 0
    pins: int = 0
    hits: int = 0
    timeouts: int = 0
    total_ttl: float = 0.0
    total_realized_utility: float = 0.0

    def update(
        self,
        ttl: float,
        actual_duration: float,
        benefit_if_hit: float,
    ) -> None:
        self.requests += 1
        self.total_ttl += ttl

        if ttl > 0:
            self.pins += 1

        hit = ttl > 0 and actual_duration <= ttl
        if hit:
            self.hits += 1
        elif ttl > 0:
            self.timeouts += 1

        # Outcome version of the paper's normalized objective.
        self.total_realized_utility += (
            benefit_if_hit if hit else 0.0
        ) - ttl

    def summary(self) -> dict[str, float]:
        n = max(1, self.requests)
        return {
            "pin_rate": self.pins / n,
            "hit_rate": self.hits / n,
            "timeout_rate": self.timeouts / n,
            "mean_ttl": self.total_ttl / n,
            "mean_realized_utility": self.total_realized_utility / n,
        }


def build_synthetic_trace(seed: int = 7) -> list[tuple[str, float]]:
    """
    Create a trace with fast, slow, long-tail, and drifting tool behavior.
    """

    rng = random.Random(seed)
    trace: list[tuple[str, float]] = []

    for i in range(360):
        selector = i % 3

        if selector == 0:
            # Distribution shift halfway through the experiment.
            mean = 0.75 if i < 180 else 1.65
            duration = max(0.05, rng.gauss(mean, 0.12))
            tool = "fast_tool"
        elif selector == 1:
            duration = max(0.05, rng.gauss(3.0, 0.30))
            tool = "slow_tool"
        else:
            duration = max(0.05, rng.lognormvariate(-0.05, 0.75))
            tool = "long_tail_tool"

        trace.append((tool, duration))

    return trace


def run_synthetic_demo(seed: int = 7) -> None:
    """
    Compare:
    1. static 2-second pin;
    2. preview repository's 0/2-second mean-threshold policy;
    3. paper-style dynamic TTL.
    """

    # Lower K only for a short demo so global/per-tool transitions appear.
    config = TTLEstimatorConfig(
        history_threshold=20,
        default_ttl_seconds=2.0,
        queue_delay_window=50,
        max_candidate_ttl_seconds=10.0,
    )
    estimator = DynamicTTLEstimator(
        config=config,
        prefill_reload_profile=ConstantPrefillReloadProfile(0.70),
    )

    # Seed queue-delay and program-length statistics.
    rng = random.Random(seed + 1)
    for _ in range(50):
        estimator.record_queue_delay(max(0.0, rng.gauss(1.40, 0.20)))
    for total_turns in [5, 6, 4, 7, 5, 6, 8, 4, 5, 7]:
        estimator.record_completed_program(total_turns)

    histories_for_preview: dict[str, list[float]] = defaultdict(list)
    metrics = {
        "static_2s": PolicyMetrics(),
        "preview_0_or_2s": PolicyMetrics(),
        "dynamic_paper_style": PolicyMetrics(),
    }
    source_counts: dict[str, int] = defaultdict(int)

    for tool, duration in build_synthetic_trace(seed):
        queue_delay = estimator.average_queue_delay()
        eta = estimator.memoryfulness()
        prefill_reload = estimator.prefill_reload_profile.estimate_seconds(2000)
        benefit_if_hit = queue_delay * eta + prefill_reload

        static_ttl = 2.0
        preview_ttl = preview_repository_ttl(
            tool,
            histories_for_preview,
            fixed_threshold_seconds=2.0,
        )
        dynamic_result = estimator.estimate_ttl(
            tool,
            context_tokens=2000,
        )

        metrics["static_2s"].update(static_ttl, duration, benefit_if_hit)
        metrics["preview_0_or_2s"].update(
            preview_ttl, duration, benefit_if_hit
        )
        metrics["dynamic_paper_style"].update(
            dynamic_result.ttl_seconds,
            duration,
            benefit_if_hit,
        )
        source_counts[dynamic_result.history_source] += 1

        # Update only after predicting, avoiding information leakage.
        estimator.record_tool_duration(tool, duration)
        histories_for_preview[tool].append(duration)

    headers = (
        "policy",
        "pin_rate",
        "hit_rate",
        "timeout_rate",
        "mean_ttl",
        "mean_utility",
    )
    rows = []
    for name, metric in metrics.items():
        summary = metric.summary()
        rows.append(
            (
                name,
                summary["pin_rate"],
                summary["hit_rate"],
                summary["timeout_rate"],
                summary["mean_ttl"],
                summary["mean_realized_utility"],
            )
        )

    widths = [max(len(headers[i]), max(len(f"{row[i]:.4f}") if i else len(row[i]) for row in rows)) for i in range(len(headers))]
    print("Synthetic TTL comparison")
    print("-" * (sum(widths) + 3 * (len(widths) - 1)))
    print(" | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    print("-" * (sum(widths) + 3 * (len(widths) - 1)))
    for row in rows:
        formatted = [row[0]]
        formatted.extend(f"{value:.4f}" for value in row[1:])
        print(" | ".join(formatted[i].ljust(widths[i]) for i in range(len(headers))))

    print("\nDynamic estimator history sources:")
    for source, count in sorted(source_counts.items()):
        print(f"  {source}: {count}")


def run_self_test() -> None:
    config = TTLEstimatorConfig(
        history_threshold=2,
        default_ttl_seconds=2.0,
    )
    estimator = DynamicTTLEstimator(
        config=config,
        prefill_reload_profile=ConstantPrefillReloadProfile(1.0),
    )

    assert estimator.estimate_ttl(None).ttl_seconds == 0.0
    assert estimator.estimate_ttl("grep").history_source == "fixed_cold_start"

    estimator.record_tool_duration("grep", 0.5)
    estimator.record_tool_duration("cat", 3.0)
    assert estimator.estimate_ttl("grep").history_source == "fixed_cold_start"

    estimator.record_tool_duration("cat", 2.5)
    assert estimator.estimate_ttl("grep").history_source == "global_history"

    estimator.record_tool_duration("grep", 0.6)
    estimator.record_tool_duration("grep", 0.7)
    result = estimator.estimate_ttl(
        "grep",
        queue_delay_seconds=1.0,
        memoryfulness=1.0,
        prefill_reload_cost_seconds=1.0,
    )
    assert result.history_source == "per_tool_history"
    assert result.ttl_seconds >= 0.0

    print("Self-test passed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone Continuum-style dynamic TTL estimator"
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run lightweight correctness checks",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="run a synthetic policy comparison",
    )
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.self_test:
        run_self_test()

    if args.demo or not args.self_test:
        run_synthetic_demo(seed=args.seed)


if __name__ == "__main__":
    main()
