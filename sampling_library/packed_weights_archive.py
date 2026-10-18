"""Lossless compact container for the experimental packed-weight model.

A JSON header describes typed views into one byte buffer. No pickle, dense
weight reconstruction or additional quantization. Tensor values are unchanged.
"""
import hashlib
import json
import math
import struct
import sys
from pathlib import Path
import torch

MAGIC=b'S1PW0001'
DTYPES={'uint8':torch.uint8,'float16':torch.float16}


def write_archive(path,bundle):
    if sys.byteorder!='little':
        raise ValueError('Archive format currently supports little-endian hosts')
    header={k:v for k,v in bundle.items() if k!='entries'}
    entries={}; data=bytearray()
    for name,entry in bundle['entries'].items():
        meta={}
        for key,value in entry.items():
            if not isinstance(value,torch.Tensor):
                meta[key]=value
                continue
            dtype=str(value.dtype).split('.')[-1]
            if dtype not in DTYPES: raise ValueError(dtype)
            # Align FP16 views while keeping the header independent of platform ABI.
            data.extend(b'\0'*(-len(data)%value.element_size()))
            payload=value.detach().cpu().contiguous().numpy().tobytes()
            meta[key]=dict(dtype=dtype,shape=list(value.shape),offset=len(data),nbytes=len(payload))
            data.extend(payload)
        entries[name]=meta
    header.update(entries=entries,byteorder='little',data_sha256=hashlib.sha256(data).hexdigest())
    encoded=json.dumps(header,separators=(',',':'),sort_keys=True,allow_nan=False).encode()
    with Path(path).open('xb') as f:
        f.write(MAGIC); f.write(struct.pack('<Q',len(encoded))); f.write(encoded); f.write(data)


def read_archive(path):
    if sys.byteorder!='little': raise ValueError('Little-endian host required')
    with Path(path).open('rb') as f:
        if f.read(len(MAGIC))!=MAGIC: raise ValueError('Invalid archive magic')
        raw=f.read(8)
        if len(raw)!=8: raise ValueError('Truncated archive header')
        length=struct.unpack('<Q',raw)[0]
        if length>Path(path).stat().st_size-len(MAGIC)-8: raise ValueError('Truncated archive')
        header=json.loads(f.read(length)); data=bytearray(f.read())
    if header.pop('byteorder')!='little': raise ValueError('Unsupported byte order')
    if hashlib.sha256(data).hexdigest()!=header.pop('data_sha256'): raise ValueError('Archive data checksum mismatch')
    for entry in header['entries'].values():
        for key in ('data','scale'):
            if key not in entry: continue
            meta=entry[key]; dtype=DTYPES[meta['dtype']]
            count=math.prod(meta['shape']); size=torch.empty((),dtype=dtype).element_size()
            offset=meta['offset']
            if count<=0 or offset<0 or count*size!=meta['nbytes'] or offset+meta['nbytes']>len(data):
                raise ValueError('Invalid tensor view')
            entry[key]=torch.frombuffer(data,dtype=dtype,count=count,offset=offset).reshape(meta['shape'])
    return header
