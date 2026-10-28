# Build vs reuse: the register and its runner

Where we wrote scientific code ourselves instead of using an open-source, peer-reviewed tool, and
what we do about it. The aim: lean on peer-reviewed methods wherever we can, and where we keep our
own code, run it in tandem with a peer-reviewed tool so the two can be compared.

This is about code, not papers: a filter, a spectrum estimate, a model, a statistic. Whether a
method's maths came from a paper is the scaffold's calculator records' job.

## Four states

| State | Meaning |
|---|---|
| reused | an open-source peer-reviewed package does the job (found automatically in dependency files and imports) |
| tandem | our own code, with a test that runs it against an open-source peer-reviewed package |
| candidate | our own code, with a known alternative that is not wired in yet |
| none-found | our own code, no alternative found yet: the search targets |
| replaced | was custom, now uses the package (kept for the record) |

"Peer-reviewed" means the package has a methods paper in a journal or reviewed proceedings, or passed
JOSS review. The catalogue (`config/reuse-catalogue.default.toml`, plus your own in the gitignored
`config/reuse-catalogue.toml`) lists the ones the runner knows, each with its paper.

## The register: `reuse.toml`

One file at the root of each repository, next to the code, so a change that adds custom code
updates it in the same pull request. Only custom code is listed; reuse is found automatically.

```toml
[[custom]]
name = "EEG quality check"
path = "apps/server/quality.py"     # file or folder, relative to the repo root
does = "band power, Goertzel tone power, a signal-quality verdict"
state = "candidate"                 # tandem | candidate | none-found | replaced
alternative = "mne"                 # a catalogue key, or a plain name not yet in the catalogue
check = ""                          # tandem: the test that compares ours with the alternative
note = "MNE psd_array_welch on the same windows"
```

Until a repository carries its own `reuse.toml`, the runner reads a seed register from
`$REUSE_HOME/registers/<owner>__<name>.toml` (default `~/.local/share/research-rag/reuse`). Seeds are
private and local; they move into each repository by pull request.

## The runner

```bash
uv run research-reuse                 # scan, write the report, add a row to the history
uv run research-reuse --check-dois    # also confirm each catalogue DOI resolves
```

It finds repositories in two places: local checkouts under `~/Sites` (`--sites`), one per GitHub
remote, and the organisation's other repositories on GitHub (`--org polarizetech`, archived ones
skipped), shallow-cloned into `$REUSE_HOME/clones`. No AI is used. For each repository it:

1. finds catalogue packages in `pyproject.toml`, `requirements*.txt`, `package.json` and imports;
2. reads the register and checks each entry: the path still exists; a tandem entry's test exists and
   mentions the alternative; whether a candidate's alternative is already a dependency of the repo
   (an easy swap);
3. writes `report.json`, `report.md` and `report.html` to `$REUSE_HOME`, and appends one row per
   repository to `history.csv`, the running tally.

Reports name private repositories and stay in `$REUSE_HOME`; never commit them to this public repo.

## What the runner cannot do

It counts what the register declares. New custom code nobody registered is found by an occasional
AI sweep (ask the gateway to review a repository against this document), and alternatives for
`none-found` entries are found by asking the gateway (`ask`) for open-source tools that do the job.
