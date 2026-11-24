# Batches on a rented server

For running a batch of questions now and then without using this machine: `cloud/batch.py` rents a
DigitalOcean droplet, runs the questions with the same pipeline, copies the results back and
destroys the droplet. A batch costs a few cents of server time plus the model's tokens.

```bash
# questions.txt: one question per line; blank lines and # comments are skipped
uv run --with boto3 python cloud/batch.py questions.txt
uv run --with boto3 python cloud/batch.py questions.txt --size s-8vcpu-16gb-amd   # faster checking
```

## What happens

1. **Index up.** A consistent snapshot of this machine's passage index (SQLite backup, safe while
   the MCP servers have it open) is uploaded to the paper library's Spaces bucket at
   `pipeline-cloud/index/passages.sqlite`. Unchanged since the last batch: not uploaded again.
2. **Droplet.** A throwaway SSH key and a droplet (default `s-4vcpu-16gb-amd`, $0.125 an hour) in the
   bucket's region, with Docker from cloud-init, tagged `research-pipeline-batch`.
3. **Code and secrets.** This checkout's files (tracked and new, never `.gitignore`d ones such as
   `config/rag.env`), the private agent briefs at their `HEAD`, the questions, and an env file
   with the endpoint key and the Spaces credentials, copied over SSH. The env file is deleted
   once the container has finished; nothing secret is built into the image.
4. **Run.** `cloud/Dockerfile` is built on the droplet and `cloud/entrypoint.sh` runs: Ollama
   pulls bge-m3 and bespoke-minicheck, the index is downloaded, and each question runs through
   `research-pipeline ask` with the cloud defaults (Qwen3.8 27B through OpenRouter, zero data
   retention, reasoning off, MiniCheck verifying on the droplet, agent briefs routed as on the
   Mac). Papers it fetches go to the same bucket, as on the Mac.
5. **Results back.** Each run's folder lands in `runs/` beside local runs, with a summary in
   `runs/cloud-batches/<batch>.json` (question, claims kept, specialist, cost, papers used). A
   copy stays in the bucket under `pipeline-cloud/batches/<batch>/`.
6. **Index down, without overwriting.** The papers the batch used are indexed into this machine's
   own index with `paper-fetch index <ids>`. The index file is never copied back: it is a cache
   built from the bucket, and replacing it under running servers could corrupt it.
7. **Teardown, always.** The droplet and the SSH key are deleted even when a step fails, and the
   script then checks that no droplet tagged `research-pipeline-batch` is left. If it reports one,
   delete it in the DigitalOcean dashboard: a powered-off droplet is still billed.

## Setup

- A DigitalOcean API token in `~/.config/polarize/do-api.token` (`chmod 600`), or
  `DIGITALOCEAN_TOKEN`, with read, create and delete on droplets and SSH keys.
- The endpoint key as for local runs (`PIPELINE_ENDPOINT_KEY_FILE` in `config/rag.env`).
- paper-fetch's Spaces settings in `~/.config/paper-fetch/library.env`.
- The private agent briefs at `~/Sites/polarize-agents` (optional; public briefs are built in).

## Cost and speed

Server time is billed per second (60-second minimum): about 12.5 cents an hour for the default
16 GB size (the script checks the region offers the size and reports the cost from DigitalOcean's
price). 8 GB is too small: the verifier, the embedding model and the paper library together ran it
out of memory.
Building the image and pulling the models takes roughly 5 to 10 minutes of each batch. Claim
checking runs on CPU there, slower than on Apple silicon, so a question can take two to four times
as long as on the Mac; a larger size shortens it. The model's tokens cost what they cost locally,
about 1 to 2 cents a question for Qwen3.8 27B.

## Limits

- Two writers can add papers to the bucket at once (the Mac and a batch). paper-fetch's per-paper
  provenance records are authoritative and its catalogue is derived, so a lost catalogue line can
  slow a lookup but never loses a paper (`paper-fetch rebuild-index` restores the catalogue).
- The pipeline's note cache (`PIPELINE_DATA_DIR`) starts empty on each batch.
- Datasets, patents and the research repository are not available to cloud runs; their sections
  are left out of those answers.
