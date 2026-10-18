import json, subprocess, sys, shutil, os
jobs = json.load(open(sys.argv[1]))
os.makedirs("/persist/runs", exist_ok=True); os.makedirs("/persist/logs", exist_ok=True)
for j in jobs:
    name = j.pop("name")
    if os.path.exists(f"/persist/runs/{name}/result.json"):
        continue
    subprocess.run([sys.executable, "/work/src/train.py", "--config", json.dumps(j), "--out", f"/work/runs/{name}"],
                   stdout=open(f"/work/logs/{name}.log", "a"), stderr=subprocess.STDOUT, cwd="/work/src")
    os.makedirs(f"/persist/runs/{name}", exist_ok=True)
    for fn in os.listdir(f"/work/runs/{name}"):
        try:
            shutil.copyfile(f"/work/runs/{name}/{fn}", f"/persist/runs/{name}/{fn}")
        except Exception as e:
            print("persist copy failed", name, fn, e, flush=True)
    try:
        shutil.copyfile(f"/work/logs/{name}.log", f"/persist/logs/{name}.log")
    except Exception as e:
        print("persist log copy failed", e, flush=True)
print("QUEUE_DONE", flush=True)
