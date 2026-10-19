#!/usr/bin/env python3
"""Copy a snapshot to a new directory, expanding lossless evidence compression."""
import argparse,gzip,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage');p.add_argument('destination',type=Path);a=p.parse_args()
r=Path(__file__).resolve().parent;root=r.parents[2];m=json.loads((r/'IMPORT.json').read_text());rows=[x for x in m['files'] if x['stage']==a.stage]
if not rows:p.error('Unknown stage')
if a.destination.exists():p.error('Destination must not exist; use a fresh disposable directory')
a.destination.mkdir(parents=True)
for row in rows:
 raw=(root/row['path']).read_bytes()
 if row.get('compression')=='gzip':raw=gzip.decompress(raw)
 assert hashlib.sha256(raw).hexdigest()==row['sha256']
 rel=Path(*Path(row['archive_member']).parts[1:]);out=a.destination/rel;out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(raw)
print('Restored',len(rows),'source/evidence files. Excluded binaries and checkpoint/data dependencies are not restored. Read IMPORT.json and the original README before reproduction.')
