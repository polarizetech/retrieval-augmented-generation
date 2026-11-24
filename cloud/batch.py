"""Run a batch of research questions on a temporary DigitalOcean droplet, then destroy it.

    uv run --with boto3 python cloud/batch.py questions.txt
    uv run --with boto3 python cloud/batch.py questions.txt --size s-8vcpu-16gb-amd  # faster

Steps (docs/cloud.md has the why):
  1. snapshot this machine's passage index and upload it to the Spaces bucket (skipped if unchanged)
  2. create a throwaway SSH key and a droplet in the bucket's region, with Docker installed
  3. copy this checkout, the private agent briefs, the questions and the secrets to it
  4. build the image and run the batch (cloud/entrypoint.sh)
  5. copy the runs back into runs/, and index the papers the batch fetched into the local index
  6. ALWAYS destroy the droplet and the SSH key, and check that nothing is left running

Needs: a DigitalOcean API token in ~/.config/polarize/do-api.token (or DIGITALOCEAN_TOKEN), with
read/create/delete on droplets and SSH keys; the endpoint key from config/rag.env; paper-fetch's
Spaces settings (~/.config/paper-fetch/library.env).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import index_sync  # noqa: E402

from research_pipeline.config import Settings, load_env  # noqa: E402

API = "https://api.digitalocean.com/v2"
TAG = "research-pipeline-batch"
TOKEN_FILE = Path.home() / ".config/polarize/do-api.token"
PRIVATE_AGENTS = Path.home() / "Sites/polarize-agents"
LOCAL_INDEX = Path(
    os.environ.get("PAPER_FETCH_INDEX") or Path.home() / ".local/share/paper-fetch/passages.sqlite"
)
PAPER_FETCH = Path.home() / ".local/bin/paper-fetch"
PREFIX = "pipeline-cloud"
USER_DATA = """#cloud-config
package_update: true
packages: [docker.io]
"""


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


class DigitalOcean:
    def __init__(self, token: str):
        self.token = token

    def call(self, method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        req = urllib.request.Request(
            API + path,
            method=method,
            data=json.dumps(body).encode() if body is not None else None,
            headers={"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read()[:300].decode(errors="replace")
            raise RuntimeError(f"DigitalOcean {method} {path}: HTTP {exc.code} {detail}") from exc
        return json.loads(raw) if raw else {}


def token() -> str:
    value = os.environ.get("DIGITALOCEAN_TOKEN", "").strip()
    if not value and TOKEN_FILE.exists():
        value = TOKEN_FILE.read_text().strip()
    if not value:
        raise SystemExit(f"no DigitalOcean API token: put one in {TOKEN_FILE} (chmod 600)")
    return value


def ssh_args(key: Path) -> list[str]:
    return [
        "-i", str(key),
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "LogLevel=ERROR",
        "-o", "ConnectTimeout=10",
        "-o", "ServerAliveInterval=30",
    ]  # fmt: skip


def run(cmd: list[str], *, check: bool = False, **kw: Any) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, check=check, **kw)  # noqa: S603 -- fixed commands


def secrets_file(directory: Path) -> Path:
    """Secrets for the container, never written into the image or the repository."""
    load_env()
    index_sync._load_env()
    settings = Settings()
    if not settings.endpoint_key:
        raise SystemExit("no endpoint key: set PIPELINE_ENDPOINT_KEY_FILE in config/rag.env")
    lines = {"PIPELINE_ENDPOINT_KEY": settings.endpoint_key}
    for key, value in os.environ.items():
        if key.startswith("PAPER_FETCH_S3_") or key in (
            "PAPER_FETCH_EMAIL", "NCBI_API_KEY", "CORE_API_KEY", "OPENALEX_API_KEY"
        ):  # fmt: skip
            lines[key] = value
    path = directory / "secrets.env"
    path.write_text("".join(f"{k}={v}\n" for k, v in lines.items()))
    path.chmod(0o600)
    return path


def bundle(directory: Path) -> None:
    """This checkout's tracked files as they are now, plus the private briefs at their HEAD."""
    # Tracked and new files, as they are on disk; .gitignore'd ones (secrets, data) never.
    listing = ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"]
    files = run(listing, cwd=ROOT, capture_output=True, check=True).stdout
    run(["tar", "--no-xattrs", "--null", "-T", "-", "-cf", str(directory / "rag.tar")],
        cwd=ROOT, input=files, check=True)  # fmt: skip
    if (PRIVATE_AGENTS / ".git").exists():
        run(["git", "archive", "-o", str(directory / "private-agents.tar"), "HEAD"],
            cwd=PRIVATE_AGENTS, check=True)  # fmt: skip


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("questions", type=Path)
    # 16 GB: the verifier (bespoke-minicheck, ~5 GB loaded), bge-m3 and the paper library with its
    # reranker do not fit in 8 GB; the model server was killed for memory there.
    parser.add_argument("--size", default="s-4vcpu-16gb-amd")
    parser.add_argument("--image", default="ubuntu-24-04-x64")
    parser.add_argument("--region", default="")
    args = parser.parse_args()
    questions = [
        q.strip() for q in args.questions.read_text().splitlines()
        if q.strip() and not q.strip().startswith("#")
    ]  # fmt: skip
    if not questions:
        raise SystemExit(f"no questions in {args.questions}")
    index_sync._load_env()
    region = args.region or os.environ.get("PAPER_FETCH_S3_REGION", "sfo3")
    batch_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    do = DigitalOcean(token())
    hourly = size_price(do, args.size, region)
    log(f"batch {batch_id}: {len(questions)} question(s) on {args.size} in {region} (${hourly}/hr)")

    work = Path(tempfile.mkdtemp(prefix="cloud-batch-"))
    snap = work / "passages.sqlite"
    index_sync.snapshot(LOCAL_INDEX, snap)
    index_sync.upload(snap, f"{PREFIX}/index/passages.sqlite")
    snap.unlink()

    key = work / "id_ed25519"
    run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", f"{TAG}-{batch_id}", "-f", str(key)],
        check=True)  # fmt: skip
    secrets = secrets_file(work)
    bundle(work)
    (work / "questions.txt").write_text("\n".join(questions) + "\n")

    key_id = droplet_id = None
    summary: Path | None = None
    started = time.time()
    outcome = 1
    try:
        key_id = do.call("POST", "/account/keys", {
            "name": f"{TAG}-{batch_id}", "public_key": key.with_suffix(".pub").read_text().strip(),
        })["ssh_key"]["id"]  # fmt: skip
        droplet_id = do.call("POST", "/droplets", {
            "name": f"{TAG}-{batch_id}".lower(), "region": region, "size": args.size,
            "image": args.image, "ssh_keys": [key_id], "tags": [TAG], "user_data": USER_DATA,
            "monitoring": False, "ipv6": False,
        })["droplet"]["id"]  # fmt: skip
        log(f"droplet {droplet_id} created; waiting for it to boot")
        ip = wait_for_ip(do, droplet_id)
        target = f"root@{ip}"
        wait_for_ssh(key, target)
        log("waiting for Docker to install")
        remote(key, target, "cloud-init status --wait >/dev/null; docker --version")

        log("copying the checkout, briefs, questions and secrets")
        remote(key, target, "mkdir -p /root/job/rag /root/job/private /root/job/batch /root/work")
        copy = ["scp", *ssh_args(key), str(work / "rag.tar"), str(secrets),
                str(work / "questions.txt")]  # fmt: skip
        if (work / "private-agents.tar").exists():
            copy.append(str(work / "private-agents.tar"))
        run([*copy, f"{target}:/root/job/"], check=True)
        remote(key, target, (
            "cd /root/job && tar -xf rag.tar -C rag && mv questions.txt batch/ && "
            "if [ -f private-agents.tar ]; then tar -xf private-agents.tar -C private; fi"
        ))  # fmt: skip

        log("building the image (a few minutes)")
        remote(key, target, "cd /root/job/rag && docker build -q -f cloud/Dockerfile -t rp .")
        log("running the batch")
        mounts = "-v /root/job/batch:/batch -v /root/work:/work"
        if (work / "private-agents.tar").exists():
            mounts += " -v /root/job/private:/agents/private:ro"
        outcome = remote(key, target, (
            f"docker run --rm --env-file /root/job/secrets.env -e BATCH_ID={batch_id} "
            f"{mounts} rp; status=$?; rm -f /root/job/secrets.env; exit $status"
        ), check=False)  # fmt: skip

        log("copying results back")
        local_runs = Settings().runs_dir
        local_runs.mkdir(parents=True, exist_ok=True)
        run(["scp", "-r", *ssh_args(key), f"{target}:/root/work/runs/*", str(local_runs)],
            check=False)  # fmt: skip
        summary_dir = local_runs / "cloud-batches"
        summary_dir.mkdir(exist_ok=True)
        summary = summary_dir / f"{batch_id}.json"
        run(["scp", *ssh_args(key), f"{target}:/root/work/batch.json", str(summary)], check=False)
        if not summary.exists():
            summary = None
    finally:
        seconds = time.time() - started
        destroy(do, droplet_id, key_id)
        secrets.unlink(missing_ok=True)

    report(summary, seconds, hourly)
    index_new_papers(summary)
    return outcome


def size_price(do: DigitalOcean, size: str, region: str) -> float:
    """The size's hourly price, after checking the region offers it (before anything is made)."""
    sizes = do.call("GET", "/sizes?per_page=200")["sizes"]
    match = next((s for s in sizes if s["slug"] == size), None)
    if match is None or region not in match["regions"] or not match["available"]:
        offered = sorted(
            s["slug"] for s in sizes
            if region in s["regions"] and s["available"] and 7000 <= s["memory"] <= 17000
        )  # fmt: skip
        raise SystemExit(f"{size} is not offered in {region}; 8-16 GB sizes there: {offered}")
    return float(match["price_hourly"])


def wait_for_ip(do: DigitalOcean, droplet_id: int) -> str:
    for _ in range(90):
        droplet = do.call("GET", f"/droplets/{droplet_id}")["droplet"]
        public = [n["ip_address"] for n in droplet["networks"]["v4"] if n["type"] == "public"]
        if droplet["status"] == "active" and public:
            return str(public[0])
        time.sleep(5)
    raise RuntimeError("droplet did not become active in 7.5 minutes")


def wait_for_ssh(key: Path, target: str) -> None:
    for _ in range(60):
        if run(["ssh", *ssh_args(key), target, "true"], capture_output=True).returncode == 0:
            return
        time.sleep(5)
    raise RuntimeError("could not reach the droplet over SSH in 5 minutes")


def remote(key: Path, target: str, command: str, check: bool = True) -> int:
    done = run(["ssh", *ssh_args(key), target, command])
    if check and done.returncode != 0:
        raise RuntimeError(f"remote command failed ({done.returncode}): {command[:80]}")
    return done.returncode


def destroy(do: DigitalOcean, droplet_id: int | None, key_id: int | None) -> None:
    """Delete the droplet and key, then confirm no tagged droplet is left billing."""
    if droplet_id is not None:
        try:
            do.call("DELETE", f"/droplets/{droplet_id}")
            log(f"droplet {droplet_id} destroyed")
        except RuntimeError as exc:
            log(f"COULD NOT DESTROY droplet {droplet_id}: {exc} -- delete it in the dashboard")
    if key_id is not None:
        try:
            do.call("DELETE", f"/account/keys/{key_id}")
        except RuntimeError as exc:
            log(f"could not delete SSH key {key_id}: {exc}")
    left: list[dict[str, Any]] = []
    for _ in range(12):
        left = do.call("GET", f"/droplets?tag_name={TAG}").get("droplets", [])
        if not left:
            log("checked: no batch droplet is running")
            return
        time.sleep(5)
    names = ", ".join(f"{d['id']} ({d['status']})" for d in left)
    log(f"WARNING: batch droplet(s) still listed: {names} -- check the DigitalOcean dashboard")


def report(summary: Path | None, seconds: float, hourly: float) -> None:
    rows = json.loads(summary.read_text()) if summary else []
    model = sum(r.get("model_cost_usd") or 0 for r in rows)
    server = seconds / 3600 * hourly
    for r in rows:
        log(f"  {r['claims_kept']}/{r['claims_drafted']} claims kept, {r['seconds']:.0f}s, "
            f"{r['specialist'] or 'researcher'}: {r['question'][:70]}")  # fmt: skip
    cost = f"server ~${server:.3f} + model ${model:.3f}"
    log(f"{len(rows)} run(s) in {seconds / 60:.1f} min; {cost}")
    if summary:
        log(f"summary: {summary}")


def index_new_papers(summary: Path | None) -> None:
    """The batch saved new papers to the bucket; index them into this machine's own index."""
    if summary is None or not PAPER_FETCH.exists():
        return
    works = sorted({w for r in json.loads(summary.read_text()) for w in r.get("papers", [])})
    if works:
        log(f"indexing {len(works)} paper(s) the batch used into the local index")
        run([str(PAPER_FETCH), "index", *works], check=False, capture_output=True)


if __name__ == "__main__":
    raise SystemExit(main())
