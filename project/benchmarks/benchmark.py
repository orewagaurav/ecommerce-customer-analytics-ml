"""Latency and memory benchmark: legacy CSV path vs feature-store path.

Produces the numbers quoted in the README. Run:

    python project/benchmarks/benchmark.py
"""

from __future__ import annotations

import gc
import json
import statistics
import sys
import time
import tracemalloc
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
if str(PROJECT_ROOT / "benchmarks") not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import pandas as pd

from src.config import get_settings
from src.feature_store import FeatureStore
from src.registry import ModelRegistry
from src.services.prediction_service import PredictionService


def _percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(len(ordered) * pct))
    return ordered[index]


def benchmark_legacy(customer_ids: list[int], settings, runs: int) -> dict:
    """The original path: re-read the CSV and re-aggregate on every call.

    Requires the processed CSV, which is no longer tracked in git. Returns None
    when it is absent so the benchmark still runs on a fresh clone.
    """
    from benchmarks.legacy_baseline import predict_customer_legacy

    if not settings.data_path.exists():
        return None

    timings: list[float] = []
    for customer_id in customer_ids[:runs]:
        started = time.perf_counter()
        predict_customer_legacy(customer_id, settings.data_path, settings.model_path)
        timings.append((time.perf_counter() - started) * 1000)

    return _summarise("legacy_csv", timings)


def benchmark_feature_store(customer_ids: list[int], settings, runs: int) -> dict:
    """The current path: warm service, O(1) feature lookup."""
    store = FeatureStore(settings.feature_store_path)
    service = PredictionService(store, settings.model_path, ModelRegistry(settings.registry_path))
    service.warm_up()

    timings: list[float] = []
    for index in range(runs):
        customer_id = customer_ids[index % len(customer_ids)]
        started = time.perf_counter()
        service.predict(customer_id)
        timings.append((time.perf_counter() - started) * 1000)

    return _summarise("feature_store", timings)


def benchmark_feature_store_no_shap(customer_ids: list[int], settings, runs: int) -> dict:
    """Scoring without SHAP, which dominates per-request cost."""
    store = FeatureStore(settings.feature_store_path)
    service = PredictionService(store, settings.model_path, ModelRegistry(settings.registry_path))
    service.warm_up()

    timings: list[float] = []
    for index in range(runs):
        customer_id = customer_ids[index % len(customer_ids)]
        started = time.perf_counter()
        service.predict(customer_id, include_explanations=False)
        timings.append((time.perf_counter() - started) * 1000)

    return _summarise("feature_store_no_shap", timings)


def _summarise(name: str, timings: list[float]) -> dict:
    return {
        "name": name,
        "runs": len(timings),
        "mean_ms": round(statistics.mean(timings), 2),
        "median_ms": round(statistics.median(timings), 2),
        "p95_ms": round(_percentile(timings, 0.95), 2),
        "min_ms": round(min(timings), 2),
        "max_ms": round(max(timings), 2),
    }


def measure_startup(settings) -> dict:
    """Cold-start cost of each data access strategy."""
    gc.collect()

    csv_seconds, csv_mb, csv_file_mb = None, None, None
    if settings.data_path.exists():
        started = time.perf_counter()
        csv_frame = pd.read_csv(settings.data_path)
        csv_seconds = round(time.perf_counter() - started, 3)
        csv_mb = round(csv_frame.memory_usage(deep=True).sum() / 1024**2, 1)
        csv_file_mb = round(settings.data_path.stat().st_size / 1024**2, 1)
        del csv_frame
        gc.collect()

    tracemalloc.start()
    started = time.perf_counter()
    store = FeatureStore(settings.feature_store_path)
    store.frame()
    store_seconds = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "csv_load_seconds": csv_seconds,
        "csv_memory_mb": csv_mb,
        "csv_file_mb": csv_file_mb,
        "feature_store_load_seconds": round(store_seconds, 3),
        "feature_store_peak_mb": round(peak / 1024**2, 1),
        "feature_store_file_mb": round(
            settings.feature_store_path.stat().st_size / 1024**2, 3
        ),
    }


def main() -> None:
    settings = get_settings()
    store = FeatureStore(settings.feature_store_path)
    customer_ids = store.customer_ids()[:60]

    print("Measuring startup and memory...")
    startup = measure_startup(settings)

    print("Benchmarking feature-store path (60 runs)...")
    fast = benchmark_feature_store(customer_ids, settings, runs=60)

    print("Benchmarking feature-store path without SHAP (60 runs)...")
    fast_no_shap = benchmark_feature_store_no_shap(customer_ids, settings, runs=60)

    # Legacy path re-reads 80 MB per call, so a small sample is enough.
    print("Benchmarking legacy CSV path (5 runs, slow by construction)...")
    legacy = benchmark_legacy(customer_ids, settings, runs=5)
    if legacy is None:
        print("  skipped: processed CSV absent (run scripts/get_data.py to compare)")

    speedup = round(legacy["mean_ms"] / fast["mean_ms"], 1) if legacy else None
    results = {
        "startup": startup,
        "benchmarks": [entry for entry in (legacy, fast, fast_no_shap) if entry],
        "speedup_vs_legacy": speedup,
    }

    output = Path(__file__).parent / "results.json"
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n" + "=" * 66)
    print(f"{'path':<26}{'mean':>10}{'median':>10}{'p95':>10}")
    print("-" * 66)
    for entry in results["benchmarks"]:
        print(
            f"{entry['name']:<26}{entry['mean_ms']:>9.1f}ms"
            f"{entry['median_ms']:>9.1f}ms{entry['p95_ms']:>9.1f}ms"
        )
    print("=" * 66)
    if speedup:
        print(f"Speed-up vs legacy CSV path: {speedup}x")
        print(f"Data file: {startup['csv_file_mb']} MB -> {startup['feature_store_file_mb']} MB")
    else:
        print(f"Feature store: {startup['feature_store_file_mb']} MB")
    print(f"Results written to {output}")


if __name__ == "__main__":
    main()
