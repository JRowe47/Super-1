# Runtime replay commands

These are limited CPU smoke checks, not training or benchmark reproduction. The source-mounted command requires the approved recovery mount layout; the exported command was tested after extracting the archive into /tmp/runtime-restore on a Linux x86-64 Debian 12 host. Do not run Linux binaries directly on macOS, extract over the host root, or provision a VM without approval.

## Original snapshot environment, read-only

```sh
sudo unshare --mount --fork /bin/sh -c 'mount --make-rprivate / && mount -t tmpfs -o size=256m,mode=1777 tmpfs /mnt/recovery-training2/tmp && chroot --userspec=1001:1002 /mnt/recovery-training2 /usr/bin/env -i PATH=/usr/bin:/bin HOME=/tmp OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/rohanarun/funcweights/representation-venv/bin/python -I -B -c '\''import json,sys,torch; torch.set_num_threads(1); x=torch.tensor([1.,2.]); y=(x*x).sum().item(); assert y==5.; print(json.dumps({'\''\'\'''\''status'\''\'\'''\'':'\''\'\'''\''PASS_RECOVERED_RUNTIME_CPU_SMOKE'\''\'\'''\'','\''\'\'''\''python'\''\'\'''\'':sys.version,'\''\'\'''\''torch'\''\'\'''\'':torch.__version__,'\''\'\'''\''cuda_build'\''\'\'''\'':torch.version.cuda,'\''\'\'''\''device'\''\'\'''\'':'\''\'\'''\''cpu'\''\'\'''\'','\''\'\'''\''sum_of_squares'\''\'\'''\'':y,'\''\'\'''\''training'\''\'\'''\'':False,'\''\'\'''\''dataset_read'\''\'\'''\'':False}))'\'''
```

The RAM-backed /tmp exists only in the isolated mount namespace. The source filesystem stays read-only. This corrected the initial failure caused by an unwritable temporary directory; it did not alter the installed packages.

## Exported archive environment

After downloading and checking all six part hashes, concatenate in numeric order and verify archive SHA-256 1363f48b4dff2752513c610effa9edcbb82bdafc61f2f3226f09a800c38f9ead. Inspect the tar members and extract into a fresh isolated Linux directory at /tmp/runtime-restore. The following exact command passed:

```sh
cd / && /usr/bin/env -i PATH=/usr/bin:/bin HOME=/tmp TMPDIR=/tmp OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHOME=/tmp/runtime-restore/usr PYTHONPATH=/tmp/runtime-restore/home/rohanarun/funcweights/representation-venv/lib/python3.10/site-packages /tmp/runtime-restore/usr/bin/python3.10 -B -s -c 'import json,sys,torch; torch.set_num_threads(1); x=torch.tensor([1.,2.]); value=(x*x).sum().item(); assert value==5.; print(json.dumps({'\''status'\'':'\''PASS_EXPORTED_RUNTIME_CPU_SMOKE'\'','\''python'\'':sys.version,'\''torch'\'':torch.__version__,'\''cuda_build'\'':torch.version.cuda,'\''prefix'\'':sys.prefix,'\''torch_file'\'':torch.__file__,'\''device'\'':'\''cpu'\'','\''sum_of_squares'\'':value,'\''training'\'':False,'\''dataset_read'\'':False,'\''host_system_libraries'\'':'\''Debian 12 recovery VM; not a full independent container'\''}))'
```

This uses host system libraries. It does not establish a complete independent container, CUDA execution, historical throughput or original benchmark scores. Package metadata and all archive part hashes are in hierarchy-runtime-manifest.json. Third-party licenses remain applicable. No credentials or source data are required by these smoke commands.
