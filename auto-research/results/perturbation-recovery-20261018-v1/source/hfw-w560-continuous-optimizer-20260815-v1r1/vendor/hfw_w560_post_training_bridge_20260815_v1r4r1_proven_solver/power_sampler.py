from __future__ import annotations

import math
import subprocess
import threading
import time
from decimal import Decimal, InvalidOperation
from typing import Callable, Mapping, Sequence


def query_gpu_power_draw_watts(
    physical_index: int, query_timeout_seconds: float,
) -> float:
    command = [
        "nvidia-smi", "-i", str(int(physical_index)),
        "--query-gpu=index,power.draw", "--format=csv,noheader,nounits",
    ]
    timeout = float(query_timeout_seconds)
    if not math.isfinite(timeout) or timeout <= 0:
        raise RuntimeError("GPU power query timeout must be finite and positive")
    query = subprocess.run(
        command, check=True, capture_output=True, text=True, timeout=timeout,
    )
    rows = [line.strip() for line in query.stdout.splitlines() if line.strip()]
    if len(rows) != 1:
        raise RuntimeError("isolated GPU power query returned an unexpected row count")
    try:
        index_text, power_text = (part.strip() for part in rows[0].split(",", 1))
        observed_index = int(index_text)
        power = Decimal(power_text)
    except (ValueError, InvalidOperation) as error:
        raise RuntimeError("GPU power.draw query is malformed") from error
    if observed_index != int(physical_index):
        raise RuntimeError("GPU power.draw physical index drift")
    if not power.is_finite() or power < 0:
        raise RuntimeError("GPU power.draw is nonfinite or negative")
    return float(power)


def integrate_power_samples_joules(samples: Sequence[Mapping[str, float]]) -> float:
    if len(samples) < 2:
        raise RuntimeError("at least two GPU power samples are required")
    total = Decimal(0)
    previous = samples[0]
    for current in samples[1:]:
        before = Decimal(str(previous["monotonic_seconds"]))
        after = Decimal(str(current["monotonic_seconds"]))
        if not before.is_finite() or not after.is_finite() or after <= before:
            raise RuntimeError("GPU power sample timestamps are not strictly monotonic")
        left = Decimal(str(previous["power_watts"]))
        right = Decimal(str(current["power_watts"]))
        if not left.is_finite() or not right.is_finite() or left < 0 or right < 0:
            raise RuntimeError("GPU power sample is nonfinite or negative")
        total += (after - before) * (left + right) / Decimal(2)
        previous = current
    if not total.is_finite() or total < 0:
        raise RuntimeError("integrated GPU energy is nonfinite or negative")
    return float(total)


class PeriodicGpuPowerSampler:
    def __init__(
        self, physical_index: int, sampling_interval_seconds: float,
        minimum_samples: int, *,
        query_timeout_seconds: float,
        query_power: Callable[[int], float],
        monotonic_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        interval = float(sampling_interval_seconds)
        if not math.isfinite(interval) or interval <= 0:
            raise RuntimeError("GPU power sampling interval must be finite and positive")
        if int(minimum_samples) < 2:
            raise RuntimeError("GPU power sampler minimum sample count must be at least two")
        timeout = float(query_timeout_seconds)
        if not math.isfinite(timeout) or timeout <= 0:
            raise RuntimeError("GPU power query timeout must be finite and positive")
        self.physical_index = int(physical_index)
        self.sampling_interval_seconds = interval
        self.minimum_samples = int(minimum_samples)
        self.query_timeout_seconds = timeout
        self._query_power = query_power
        self._monotonic_clock = monotonic_clock
        self._samples: list[dict[str, float]] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._sampler_error: BaseException | None = None
        self._closed = False
        self._receipt: Mapping[str, object] | None = None

    def _take_sample(self) -> None:
        power = float(self._query_power(self.physical_index))
        timestamp = float(self._monotonic_clock())
        if not math.isfinite(power) or power < 0:
            raise RuntimeError("GPU power sampler received invalid watts")
        if not math.isfinite(timestamp):
            raise RuntimeError("GPU power sampler received a nonfinite timestamp")
        with self._lock:
            if self._samples and timestamp <= self._samples[-1]["monotonic_seconds"]:
                raise RuntimeError("GPU power timestamps did not advance monotonically")
            self._samples.append({
                "monotonic_seconds": timestamp,
                "power_watts": power,
            })

    def _sample_loop(self) -> None:
        try:
            while not self._stop.wait(self.sampling_interval_seconds):
                self._take_sample()
        except BaseException as error:
            self._sampler_error = error
            self._stop.set()

    def __enter__(self) -> "PeriodicGpuPowerSampler":
        if self._thread is not None or self._closed:
            raise RuntimeError("GPU power sampler is single-use")
        self._take_sample()
        self._thread = threading.Thread(
            target=self._sample_loop,
            name=f"gpu-power-sampler-physical-{self.physical_index}",
            daemon=False,
        )
        self._thread.start()
        return self

    def close(self) -> Mapping[str, object]:
        if self._closed:
            if self._receipt is None:
                raise RuntimeError("GPU power sampler closed without a receipt")
            return self._receipt
        self._stop.set()
        if self._thread is None:
            raise RuntimeError("GPU power sampler was never started")
        self._thread.join(timeout=(
            self.query_timeout_seconds + self.sampling_interval_seconds + 5.0
        ))
        if self._thread.is_alive():
            raise RuntimeError("GPU power sampler background thread did not stop")
        self._closed = True
        if self._sampler_error is not None:
            raise RuntimeError("GPU power sampler background query failed") from self._sampler_error
        self._take_sample()
        with self._lock:
            samples = tuple(dict(sample) for sample in self._samples)
        if len(samples) < self.minimum_samples:
            raise RuntimeError("GPU power sampler did not meet its minimum sample count")
        energy = integrate_power_samples_joules(samples)
        powers = [sample["power_watts"] for sample in samples]
        self._receipt = {
            "schema": "hfw_periodic_gpu_power_draw_energy_receipt_v1",
            "status": "PASS_PERIODIC_GPU_POWER_DRAW_INTEGRATED",
            "mode": "nvidia_smi_power_draw_periodic_trapezoidal_integration",
            "physical_cuda_index": self.physical_index,
            "query_field": "power.draw",
            "power_units": "watts",
            "energy_units": "joules",
            "timestamp_units": "monotonic_seconds",
            "sampling_interval_seconds": self.sampling_interval_seconds,
            "query_timeout_seconds": self.query_timeout_seconds,
            "minimum_samples": self.minimum_samples,
            "sample_count": len(samples),
            "start_monotonic_seconds": samples[0]["monotonic_seconds"],
            "end_monotonic_seconds": samples[-1]["monotonic_seconds"],
            "observed_duration_seconds": (
                samples[-1]["monotonic_seconds"] - samples[0]["monotonic_seconds"]
            ),
            "integrated_energy_joules": energy,
            "minimum_power_watts": min(powers),
            "maximum_power_watts": max(powers),
            "mean_sampled_power_watts": sum(powers) / len(powers),
            "samples": list(samples),
            "background_thread_stopped": True,
        }
        return self._receipt

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        try:
            self.close()
        except BaseException as close_error:
            if exc_type is None:
                raise
            raise RuntimeError(
                "measured body and GPU power sampler both failed; "
                f"body={exc_type.__name__}: {exc_value}; "
                f"sampler={type(close_error).__name__}: {close_error}"
            ) from close_error
        return False

    @property
    def thread_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def receipt(self) -> Mapping[str, object]:
        if self._receipt is None:
            raise RuntimeError("GPU power sampler receipt requested before clean close")
        return self._receipt


def power_sampler_from_protocol(
    protocol: Mapping[str, object], physical_index: int,
) -> PeriodicGpuPowerSampler:
    compute = protocol["compute"]
    return PeriodicGpuPowerSampler(
        physical_index,
        float(compute["power_draw_sampling_interval_seconds"]),
        int(compute["power_draw_minimum_samples"]),
        query_timeout_seconds=float(compute["power_draw_query_timeout_seconds"]),
        query_power=lambda index: query_gpu_power_draw_watts(
            index, float(compute["power_draw_query_timeout_seconds"])
        ),
    )
