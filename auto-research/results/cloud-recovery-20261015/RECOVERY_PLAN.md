# Authorized cloud artifact recovery

User approved a temporary CPU-only recovery VM and read-only inspection of snapshot copies, followed by cleanup. No original disk, GPU instance, training process, or official test is to be modified or started.

Project: local-passage-501917-g0. Zone: us-central1-a.

Temporary resources created or requested:
- fw-recovery-cpu-20261015 (e2-standard-2, Debian 12, boot disk created at 30 GB then resized to 200 GB for temporary I/O performance)
- fw-recovery-training-20261015
- fw-recovery-training2-20261015
- fw-recovery-training3-20261015
- fw-recovery-training4-20261015

Recovery data disks are copies of the latest October 15 snapshots. Attach in READ_ONLY mode and mount ext4 with ro,noload to prevent journal replay. Do not execute training source from recovered disks or load untrusted checkpoints. A narrow installed-runtime CPU arithmetic smoke test was run as the original unprivileged UID in an isolated mount namespace with ephemeral RAM-backed /tmp; the source disk remained read-only.

Recorded nanoGPT checkpoint SHA-256: 7de4e9d81b606ec6b0d7e9fbaf4dcdcb30aa55456e775a07371aaf65a6591241.
Recorded path: /home/rohanarun/funcweights-aug8-official-nanogpt-v1-runs/qwen06b-aug8-official-nanogpt-10m-20260815T110300Z-v1-attempt2/student_final.pt.

SSD capacity is constrained. Large recovered artifacts must be streamed to private release storage, not downloaded over the remaining SSD capacity. Preserve hashes and provenance before deletion of temporary resources. Original disks and snapshots are never cleanup targets.
