import urllib.request, os
base='https://huggingface.co/datasets/kjj0/fineweb10B-gpt2/resolve/main/'
D = os.environ.get('DATA_DIR', '/work/data'); os.makedirs(D, exist_ok=True); os.chdir(D)
for n in ['fineweb_val_000000.bin']+[f'fineweb_train_{i:06d}.bin' for i in range(1,11)]:
    if os.path.exists(n) and os.path.getsize(n)>1e8: continue
    urllib.request.urlretrieve(base+n, n+'.part'); os.rename(n+'.part', n); print(n, os.path.getsize(n), flush=True)
print("DONE")
