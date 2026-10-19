import hashlib,json
from pathlib import Path
r=Path(__file__).resolve().parent
m=json.loads((r/'INVENTORY.json').read_text())
for f in m['files']:
 p=r/f['path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==f['published_sha256'],p
print('PASS:',len(m['files']),'published records; no model execution')
