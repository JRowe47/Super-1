"""Verify each checkpoint upload and publish a compact, sanitized inventory."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--archive-dir', type=Path, required=True)
    args = ap.parse_args()
    protocol = json.loads((ROOT / 'PROTOCOL.json').read_text())
    records = []
    for recipe in protocol['runs']:
        run_id = recipe['run_id']
        out = ROOT / 'execution' / run_id
        receipt = json.loads((out / 'PRESERVATION.json').read_text())
        result = receipt['upload_result']['results']
        assert len(result) == 1
        result = result[0]
        assert result['status'] == 'succeeded' and result['local_metadata_applied']
        archive = args.archive_dir / receipt['archive_name']
        assert archive.stat().st_size == receipt['archive_bytes']
        assert sha(archive) == receipt['archive_sha256']
        assert sha(out / 'checkpoint.pt') == receipt['checkpoint_sha256']
        with zipfile.ZipFile(archive) as z:
            assert z.namelist() == ['checkpoint.pt']
            assert hashlib.sha256(z.read('checkpoint.pt')).hexdigest() == receipt['checkpoint_sha256']
        records.append({
            **{key: receipt[key] for key in (
                'run_id', 'archive_name', 'archive_bytes', 'archive_sha256', 'checkpoint_sha256')},
            'preservation_status': 'succeeded',
            'library_file_id': result['library_file_id'],
            'version': result['current_version_number'],
        })
    inventory = {
        'run_id': protocol['run_id'],
        'scope': 'Contributor-held archives outside Git. Each contains checkpoint.pt with model, optimizer, RNG, recipe and complete traces.',
        'all_archive_and_checkpoint_hashes_verified': True,
        'records': records,
    }
    (ROOT / 'CHECKPOINT_ARCHIVES.json').write_text(json.dumps(inventory, indent=2) + '\n')
    terminal_path = ROOT / 'TERMINAL.json'
    if terminal_path.exists():
        terminal = json.loads(terminal_path.read_text())
        terminal.update(checkpoint_archives_verified=len(records),
                        checkpoint_uploads_require_separate_receipt_check=False)
        if (ROOT / 'ARCHIVE_REPAIRS.json').exists():
            terminal['archive_repair_provenance'] = 'ARCHIVE_REPAIRS.json'
        terminal_path.write_text(json.dumps(terminal, indent=2) + '\n')
    print(f'Preserved and hash-checked {len(records)} checkpoint archives.')


if __name__ == '__main__':
    main()
