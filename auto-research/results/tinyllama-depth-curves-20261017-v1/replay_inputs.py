"""Reacquire byte-exact published inputs, without resolving any moving refs.

Added as a reproduction helper after the measured run. Uses the frozen
acquisition implementation and the public pinned URLs/hashes in INPUTS.json.
"""
import argparse
import importlib.util
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    args=parser.parse_args()
    if args.manifest.exists():raise RuntimeError('Refusing to replace an existing input manifest')
    spec=importlib.util.spec_from_file_location('frozen_acquire',Path(__file__).with_name('acquire.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    reference=json.loads(args.reference.read_text())
    output={k:v for k,v in reference.items() if k!='files'}
    output['files']=[]
    for item in reference['files']:
        name=Path(item['path']).name
        destination=args.inputs/name if name.endswith('.parquet') else args.inputs/'model'/name
        output['files'].append(module.fetch(item['source_url'],destination,item['sha256'],item['bytes']))
    args.manifest.parent.mkdir(parents=True,exist_ok=True)
    args.manifest.write_text(json.dumps(output,indent=2)+'\n')


if __name__=='__main__':main()
