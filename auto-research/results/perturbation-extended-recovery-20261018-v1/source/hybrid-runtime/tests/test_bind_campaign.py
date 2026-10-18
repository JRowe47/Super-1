from __future__ import annotations

import argparse
import errno
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import hfw_dist.bind_campaign as bind_module
from hfw_dist.common import hash_file


class CheckpointImportTests(unittest.TestCase):
    def _paths(self, root: Path) -> tuple[Path, Path, dict[str, object]]:
        source = root / "source.pt"
        source.write_bytes(b"checkpoint-bytes" * 37)
        destination = root / "authority" / "checkpoint.pt"
        destination.parent.mkdir()
        return source, destination, hash_file(source)

    def _cross_device_link(self, source: Path, destination: Path):
        real_link = os.link

        def link(source_path, destination_path, *args, **kwargs):
            if (
                Path(source_path) == source
                and Path(destination_path) == destination
            ):
                raise OSError(errno.EXDEV, "simulated cross-device hardlink")
            return real_link(source_path, destination_path, *args, **kwargs)

        return link

    def test_cross_device_hardlink_falls_back_to_durable_atomic_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, destination, expected = self._paths(root)
            real_fsync_directory = bind_module.fsync_directory
            with (
                patch.object(
                    bind_module.os, "link",
                    side_effect=self._cross_device_link(source, destination),
                ),
                patch.object(
                    bind_module, "fsync_directory",
                    wraps=real_fsync_directory,
                ) as fsync_directory,
            ):
                installed = bind_module._import_checkpoint_create_only(
                    source, destination,
                    expected_sha256=str(expected["sha256"]),
                    expected_size_bytes=int(expected["size_bytes"]),
                )

            self.assertEqual(destination.read_bytes(), source.read_bytes())
            self.assertEqual(installed["sha256"], expected["sha256"])
            self.assertEqual(installed["size_bytes"], expected["size_bytes"])
            self.assertNotEqual(destination.stat().st_ino, source.stat().st_ino)
            self.assertGreaterEqual(fsync_directory.call_count, 1)
            self.assertFalse(list(destination.parent.glob(f".{destination.name}.*")))

    def test_fallback_publish_is_create_only_and_preserves_existing_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, destination, expected = self._paths(root)
            destination.write_bytes(b"already-present")

            with patch.object(
                bind_module.os, "link",
                side_effect=self._cross_device_link(source, destination),
            ):
                with self.assertRaises(FileExistsError):
                    bind_module._import_checkpoint_create_only(
                        source, destination,
                        expected_sha256=str(expected["sha256"]),
                        expected_size_bytes=int(expected["size_bytes"]),
                    )

            self.assertEqual(destination.read_bytes(), b"already-present")
            self.assertFalse(list(destination.parent.glob(f".{destination.name}.*")))

    def test_fallback_rejects_corrupt_staging_bytes_before_publish(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, destination, expected = self._paths(root)

            def corrupt_copy(_source_handle, destination_handle, *, length):
                self.assertEqual(length, 1024 * 1024)
                destination_handle.write(b"corrupt")

            with (
                patch.object(
                    bind_module.os, "link",
                    side_effect=self._cross_device_link(source, destination),
                ),
                patch.object(
                    bind_module.shutil, "copyfileobj", side_effect=corrupt_copy,
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
                    bind_module._import_checkpoint_create_only(
                        source, destination,
                        expected_sha256=str(expected["sha256"]),
                        expected_size_bytes=int(expected["size_bytes"]),
                    )

            self.assertFalse(destination.exists())
            self.assertFalse(list(destination.parent.glob(f".{destination.name}.*")))


class BindingRegressionTests(unittest.TestCase):
    def test_credential_scan_counts_each_payload_byte_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payloads = {
                "a.bin": b"first payload",
                "nested/b.bin": b"second payload",
            }
            for relative, payload in payloads.items():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)

            scan = bind_module._scan_snapshot_credentials(
                root, {relative: {} for relative in reversed(payloads)},
            )

            ordered_payload = b"".join(payloads[key] for key in sorted(payloads))
            self.assertEqual(scan["files_scanned"], len(payloads))
            self.assertEqual(scan["bytes_scanned"], len(ordered_payload))
            self.assertEqual(
                scan["concatenated_payload_sha256"],
                hashlib.sha256(ordered_payload).hexdigest(),
            )

    def test_main_parses_arguments_exactly_once(self):
        arguments = SimpleNamespace(
            package_root=Path("package"), source_ledger=Path("ledger"),
            workers=4, shortlist_per_group=1, nonce="nonce",
            cpu_limit=2, memory_limit_gib=4, disk_limit_gib=8,
            torch_threads=2, interop_threads=1,
            minimum_relative_verify_improvement=0.0001,
            duplicate_forward_tolerance=1e-7,
            worker_python_major_minor="3.12",
            worker_platform_system="Linux", worker_machine="x86_64",
            worker_torch_version="2.8.0+cpu", worker_numpy_version="2.0.2",
            host_mps_lock_path=Path("mps.lock"),
        )
        with (
            patch.object(
                argparse.ArgumentParser, "parse_args", return_value=arguments,
            ) as parse_args,
            patch.object(bind_module, "bind_campaign", return_value={}),
            patch("builtins.print"),
        ):
            bind_module.main()

        parse_args.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
