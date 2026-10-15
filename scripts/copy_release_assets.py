"""Copy GitHub release assets without staging multi-gigabyte files on disk.

Requires gh authentication. Credentials are held in memory only. Existing assets
are accepted only when their size and GitHub SHA-256 digest match the source.
"""
import hashlib
import http.client
import json
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

SOURCE = "rohanarun/function-weights-research"
DESTINATION = "rohanarun/Super-1"


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def main():
    token = subprocess.check_output(["gh", "auth", "token"], text=True).strip()
    headers = {"Authorization": "Bearer " + token,
               "Accept": "application/vnd.github+json", "User-Agent": "Super-1-release-copy"}

    def api(path, data=None):
        payload = None if data is None else json.dumps(data).encode()
        request = urllib.request.Request("https://api.github.com/" + path,
                                         data=payload, headers=headers)
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.load(response)

    receipts = []
    for release in api("repos/" + SOURCE + "/releases?per_page=100"):
        tag = release["tag_name"]
        try:
            target = api("repos/" + DESTINATION + "/releases/tags/" + tag)
        except urllib.error.HTTPError as error:
            if error.code != 404:
                raise
            target = api("repos/" + DESTINATION + "/releases", {
                "tag_name": tag, "target_commitish": "main",
                "name": release["name"], "body": release["body"],
                "draft": False, "prerelease": release["prerelease"]})
        existing = {asset["name"]: asset for asset in target["assets"]}
        for asset in release["assets"]:
            expected = asset.get("digest")
            if not expected or not expected.startswith("sha256:"):
                raise RuntimeError("Missing source SHA-256: " + asset["name"])
            copied = existing.get(asset["name"])
            if copied is None:
                download_headers = dict(headers, Accept="application/octet-stream")
                request = urllib.request.Request(asset["url"], headers=download_headers)
                try:
                    source = urllib.request.build_opener(NoRedirect).open(request, timeout=120)
                except urllib.error.HTTPError as redirect:
                    if redirect.code not in (301, 302, 303, 307, 308):
                        raise
                    # Never forward GitHub authorization to signed storage URLs.
                    source = urllib.request.urlopen(redirect.headers["Location"], timeout=120)
                upload = urllib.parse.urlsplit(target["upload_url"].split("{")[0])
                connection = http.client.HTTPSConnection(upload.hostname, timeout=600)
                path = upload.path + "?" + urllib.parse.urlencode({"name": asset["name"]})
                connection.putrequest("POST", path)
                connection.putheader("Authorization", "Bearer " + token)
                connection.putheader("User-Agent", "Super-1-release-copy")
                connection.putheader("Content-Type", "application/octet-stream")
                connection.putheader("Content-Length", str(asset["size"]))
                connection.endheaders()
                digest = hashlib.sha256()
                total = 0
                with source:
                    while True:
                        chunk = source.read(1024 * 1024)
                        if not chunk:
                            break
                        digest.update(chunk)
                        total += len(chunk)
                        connection.send(chunk)
                response = connection.getresponse()
                body = response.read()
                connection.close()
                if response.status != 201:
                    raise RuntimeError("Upload failed HTTP " + str(response.status))
                copied = json.loads(body)
                if total != asset["size"] or "sha256:" + digest.hexdigest() != expected:
                    raise RuntimeError("Stream verification failed: " + asset["name"])
            if copied["size"] != asset["size"] or copied.get("digest") != expected:
                raise RuntimeError("Destination mismatch: " + asset["name"])
            receipts.append({"tag": tag, "name": asset["name"], "size": asset["size"],
                             "digest": expected, "url": copied["browser_download_url"]})
            print("Verified", tag, asset["name"], asset["size"], flush=True)
    receipt = {"source": SOURCE, "destination": DESTINATION, "assets": receipts,
               "asset_count": len(receipts), "total_bytes": sum(x["size"] for x in receipts)}
    Path("RELEASE_ASSET_COPY.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print("All release assets verified:", len(receipts), flush=True)


if __name__ == "__main__":
    main()
