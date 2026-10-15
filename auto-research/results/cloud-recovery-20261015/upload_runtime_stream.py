"""Stream allowlisted runtime parts to private GitHub; never forward credentials."""
import argparse
import hashlib
import http.client
import json
import pathlib
import subprocess
import urllib.parse

root = pathlib.Path(__file__).parent
parser = argparse.ArgumentParser()
parser.add_argument('--manifest', type=pathlib.Path, default=root / 'hierarchy-runtime-manifest.json')
parser.add_argument('--tag', default='recovered-runtime-20261015')
args = parser.parse_args()
manifest = json.loads(args.manifest.read_text())
repo = 'rohanarun/function-weights-research'
tag = args.tag
release = json.loads(subprocess.check_output(['gh', 'api', f'repos/{repo}/releases/tags/{tag}']))
token = subprocess.check_output(['gh', 'auth', 'token']).decode().strip()
existing = {a['name']: a for a in release['assets']}
for part in manifest['parts']:
    name = part['name']
    if name in existing:
        assert existing[name].get('digest') == 'sha256:' + part['sha256']
        print('ALREADY_VERIFIED ' + name, flush=True)
        continue
    command = ['gcloud', 'compute', 'ssh', 'fw-recovery-cpu-20261015',
               '--project=local-passage-501917-g0', '--zone=us-central1-a', '--quiet',
               '--command=sudo cat ' + part.get('remote_path', '/tmp/' + name)]
    command[0] = '/opt/homebrew/share/google-cloud-sdk/bin/gcloud'
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    connection = http.client.HTTPSConnection('uploads.github.com', timeout=120)
    url = f'/repos/{repo}/releases/{release["id"]}/assets?name=' + urllib.parse.quote(name)
    connection.putrequest('POST', url)
    connection.putheader('Authorization', 'Bearer ' + token)
    connection.putheader('User-Agent', 'artifact-recovery')
    connection.putheader('Content-Type', 'application/octet-stream')
    connection.putheader('Content-Length', str(part['bytes']))
    connection.endheaders()
    digest = hashlib.sha256()
    count = 0
    while True:
        block = process.stdout.read(1024 * 1024)
        if not block:
            break
        count += len(block)
        digest.update(block)
        connection.send(block)
    process.wait()
    assert process.returncode == 0, 'Remote read failed; reconcile release before retry'
    assert count == part['bytes'] and digest.hexdigest() == part['sha256']
    response = connection.getresponse()
    result = json.loads(response.read())
    assert response.status == 201, 'Upload failed; reconcile release before retry'
    assert result.get('digest') == 'sha256:' + part['sha256']
    print(json.dumps({'name': name, 'size': count, 'digest': result['digest'], 'state': result['state']}), flush=True)
    connection.close()
