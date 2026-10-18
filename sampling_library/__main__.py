"""List, verify or register evidence-backed parameters: python -m sampling_library."""
import argparse
import json
from . import read, register, validate, DEFAULT_INDEX
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('command', choices=['list', 'verify', 'register'])
p.add_argument('--index', default=str(DEFAULT_INDEX))
p.add_argument('--entry', help='JSON entry for register; hashes and evidence are required')
a = p.parse_args()
if a.command == 'register':
    if not a.entry:
        p.error('--entry is required')
    register(json.load(open(a.entry)), a.index)
else:
    data = read(a.index)
    for entry in data['entries']:
        if a.command == 'verify':
            validate(entry)
        print(json.dumps({k: entry[k] for k in ('id', 'use_case', 'seed', 'status', 'active')}))
