from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Mapping

import torch


def validate_no_autograd_resource_gate(
    protocol: Mapping[str, object], output_root: Path,
    no_writer_receipt: Mapping[str, object], *,
    physical_cuda_indices: tuple[int, ...] | None = None,
) -> Mapping[str, object]:
    if no_writer_receipt.get("status") != "PASS_NO_WRITERS":
        raise RuntimeError("fresh no-writer receipt is required")
    if protocol["search"]["autograd_calls"] != 0 or protocol["search"]["optimizer_steps"] != 0:
        raise RuntimeError("protocol authorizes gradients or optimizer steps")
    compute = protocol["compute"]
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the W560 production bridge")
    device_count = torch.cuda.device_count()
    if physical_cuda_indices is None and device_count < int(compute["minimum_cuda_devices"]):
        raise RuntimeError("insufficient measured CUDA device count")
    if physical_cuda_indices is not None and device_count != len(physical_cuda_indices):
        raise RuntimeError("isolated CUDA logical/physical device-count drift")
    devices = []
    for index in range(device_count):
        with torch.cuda.device(index):
            free_gpu, total_gpu = torch.cuda.mem_get_info()
        if free_gpu < int(compute["minimum_free_gpu_bytes_per_device"]):
            raise RuntimeError(f"insufficient free GPU memory on device {index}")
        devices.append({
            "logical_index": index,
            "physical_index": (
                index if physical_cuda_indices is None else physical_cuda_indices[index]
            ),
            "name": torch.cuda.get_device_name(index),
            "free_bytes": free_gpu,
            "total_bytes": total_gpu,
        })
    command = ["nvidia-smi"]
    if physical_cuda_indices is not None:
        command.extend(["-i", ",".join(str(value) for value in physical_cuda_indices)])
    command.extend([
            "--query-gpu=index,name,memory.free,memory.total,utilization.gpu,temperature.gpu,ecc.errors.uncorrected.volatile.total",
            "--format=csv,noheader,nounits",
    ])
    query = subprocess.run(
        command, check=True, capture_output=True, text=True,
    )
    telemetry = [line.strip() for line in query.stdout.splitlines() if line.strip()]
    if len(telemetry) != device_count:
        raise RuntimeError("nvidia-smi telemetry device count drift")
    parsed_telemetry = []
    for line in telemetry:
        fields = [value.strip() for value in line.split(",")]
        if len(fields) != 7:
            raise RuntimeError("nvidia-smi telemetry field-count drift")
        try:
            row = {
                "index": int(fields[0]), "name": fields[1],
                "free_memory_mib": int(fields[2]), "total_memory_mib": int(fields[3]),
                "utilization_percent": int(fields[4]), "temperature_c": int(fields[5]),
                "volatile_uncorrected_ecc_errors": int(fields[6]),
            }
        except ValueError as error:
            raise RuntimeError("nvidia-smi telemetry is not numeric") from error
        if row["utilization_percent"] > int(compute["maximum_prelaunch_gpu_utilization_percent"]):
            raise RuntimeError(f"GPU {row['index']} is busy before launch")
        if row["temperature_c"] > int(compute["maximum_prelaunch_temperature_c"]):
            raise RuntimeError(f"GPU {row['index']} temperature is too high")
        if row["volatile_uncorrected_ecc_errors"] > int(compute["maximum_volatile_uncorrected_ecc_errors"]):
            raise RuntimeError(f"GPU {row['index']} has uncorrected volatile ECC errors")
        parsed_telemetry.append(row)
    if physical_cuda_indices is not None and tuple(
        row["index"] for row in parsed_telemetry
    ) != tuple(physical_cuda_indices):
        raise RuntimeError("isolated nvidia-smi physical index drift")
    free_disk = shutil.disk_usage(output_root.parent).free
    if free_disk < int(compute["minimum_free_output_disk_bytes"]):
        raise RuntimeError("less than 30 GiB free output disk")
    return {
        "status": "PASS_NO_AUTOGRAD_RESOURCE_GATE",
        "cuda_devices": devices,
        "nvidia_smi_telemetry": parsed_telemetry,
        "free_disk_bytes": free_disk,
        "autograd_calls": 0,
        "optimizer_steps": 0,
        "no_writer_receipt_sha256": no_writer_receipt.get("receipt_file_sha256"),
    }
