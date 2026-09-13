"""Inference latency benchmarking module using time.perf_counter()."""

import json
import logging
import platform
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from config.config import BEST_MODEL_PATH, INFERENCE_BENCHMARK_PATH, INPUT_FEATURES
from src.prediction import load_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_latency_benchmark(
    model_path: Optional[Path] = None,
    n_single_runs: int = 500,
    warmup_runs: int = 50,
    save_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute high-resolution latency benchmark on the trained model pipeline.

    Parameters
    ----------
    model_path : Path, optional
        Path to serialized model.
    n_single_runs : int, default 500
        Number of single-sample repetitions for latency measurement.
    warmup_runs : int, default 50
        Number of warmup iterations before recording metrics.
    save_path : Path, optional
        Path to save benchmark JSON results.

    Returns
    -------
    Dict[str, Any]
        Benchmarking statistics.
    """
    model = load_model(model_path)

    # Prepare standard reference sample (median-like)
    sample_dict = {
        "tau1": 2.959,
        "tau2": 3.079,
        "tau3": 8.381,
        "tau4": 9.780,
        "p1": 3.763,
        "p2": -1.527,
        "p3": -1.390,
        "p4": -0.845,
        "g1": 0.562,
        "g2": 0.413,
        "g3": 0.778,
        "g4": 0.958,
    }
    df_single = pd.DataFrame([sample_dict], columns=INPUT_FEATURES)

    logger.info("Executing benchmark warmup runs...")
    for _ in range(warmup_runs):
        _ = model.predict(df_single)

    logger.info(f"Measuring single-sample latency over {n_single_runs} runs...")
    single_latencies_ms = []
    for _ in range(n_single_runs):
        t0 = time.perf_counter()
        _ = model.predict(df_single)
        t1 = time.perf_counter()
        single_latencies_ms.append((t1 - t0) * 1000.0)

    single_latencies_arr = np.array(single_latencies_ms)
    mean_lat = float(np.mean(single_latencies_arr))
    median_lat = float(np.median(single_latencies_arr))
    p95_lat = float(np.percentile(single_latencies_arr, 95))
    p99_lat = float(np.percentile(single_latencies_arr, 99))
    min_lat = float(np.min(single_latencies_arr))
    max_lat = float(np.max(single_latencies_arr))
    throughput = float(1000.0 / mean_lat) if mean_lat > 0 else 0.0

    # Batch latency test
    batch_sizes = [1, 10, 50, 100, 500, 1000]
    batch_results = {}
    for bs in batch_sizes:
        df_batch = pd.concat([df_single] * bs, ignore_index=True)
        batch_times = []
        for _ in range(30):
            t0 = time.perf_counter()
            _ = model.predict(df_batch)
            t1 = time.perf_counter()
            batch_times.append((t1 - t0) * 1000.0)
        mean_batch_ms = float(np.mean(batch_times))
        batch_results[f"batch_{bs}"] = {
            "batch_size": bs,
            "mean_total_ms": round(mean_batch_ms, 3),
            "latency_per_sample_ms": round(mean_batch_ms / bs, 4),
            "samples_per_second": round((bs / mean_batch_ms) * 1000.0, 1),
        }

    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
        },
        "single_sample": {
            "runs": n_single_runs,
            "mean_ms": round(mean_lat, 4),
            "median_ms": round(median_lat, 4),
            "p95_ms": round(p95_lat, 4),
            "p99_ms": round(p99_lat, 4),
            "min_ms": round(min_lat, 4),
            "max_ms": round(max_lat, 4),
            "throughput_samples_per_sec": round(throughput, 1),
        },
        "batch_benchmarks": batch_results,
        "includes_preprocessing": True,
        "scientific_disclaimer": (
            "The measured inference latency demonstrates the computational feasibility of "
            "low-latency classification under the benchmark conditions."
        ),
    }

    out_file = save_path or INFERENCE_BENCHMARK_PATH
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)
    logger.info(f"Saved benchmark results to {out_file}")

    print("\n" + "=" * 50)
    print("INFERENCE BENCHMARK RESULTS")
    print("=" * 50)
    print(f"Mean Latency:    {mean_lat:.4f} ms/sample")
    print(f"Median Latency:  {median_lat:.4f} ms/sample")
    print(f"P95 Latency:     {p95_lat:.4f} ms/sample")
    print(f"Throughput:      {throughput:.1f} samples/sec")
    print(f"Includes Preprocessing: Yes (Full Pipeline)")
    print("=" * 50 + "\n")

    return results


if __name__ == "__main__":
    run_latency_benchmark()
