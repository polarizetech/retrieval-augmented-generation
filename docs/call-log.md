# The call log and record pages

Every tool call through the gateway is recorded: when, who called (client, session, project), the
tool, its arguments, the full response, and what it refers to (research runs, saved reports,
requests, DOIs, GitHub and doi.org links). The records live in a hosted Postgres database, Neon
today, so they don't depend on this Mac.

Each reply also links to its record, and to the page of any research doc it mentions. That page
shows the doc together with every call that refers to it, so you can see when and where it was
used.

Code: [`research_mcp/calllog.py`](../research_mcp/calllog.py) (recording, copying, reading) and
[`research_mcp/records.py`](../research_mcp/records.py) (the pages).

## How it works

1. **The call is written locally first.** It goes to a small SQLite file,
   `calls.sqlite` in the gateway's data folder, so a slow or unreachable database never slows down
   a tool call.
2. **The gateway copies it to Postgres right after.** It sends 200 rows at a time and retries
   every minute while the database is unreachable. Nothing is lost while it's down.
3. **The local file is only an outbox.** Rows the database already has are deleted from it after
   `CALL_LOG_LOCAL_DAYS` (7 by default). With no database configured, nothing is deleted.
4. **Replies get a `links` field.** It holds:
   - `record`: this call's page.
   - `research`: the page of each run or saved report the call mentions.
   - `requests`: the history of the `ask` request it belongs to.

   Where the reply has a `relay`, a **Links** list is added to the end of it as well.
   `ask` is recorded in full but sends the client a compact brief ([mcp-clients.md](mcp-clients.md)),
   so its record is where the full reply lives.
   Status tools (`gateway_status`, `hub__status`, `hub__board`, `hub__usage`) are recorded but get
   no links.
5. **Record pages are served by the gateway under `/records`.** They only answer requests from this
   Mac, directly or through tailnet-only `tailscale serve`. A request through Tailscale Funnel, or
   addressed to the public host, gets a 404. The public MCP host never shows them.

| Page | Shows |
|---|---|
| `/records` | The 100 most recent calls, and whether the database copy is up to date. |
| `/records/<id>` | One call: who, when, summary, references, related calls, arguments and the full response. |
| `/records/run/<folder or run id>` | A research run or saved report: its `answer.md`, its run ids, and **every call that refers to it**. |
| `/records/request/<request id>` | Every call that belongs to one `ask` request. |

`gateway_status` reports the log's state under `call_log`: whether a database is configured, how
many calls haven't been copied yet, and the last error.

### What's stored, and where

Your questions, the tools' arguments and full responses (including passage text) are stored in
the database, on Neon's servers. Credentials never pass through tool arguments, so none are
stored.

### Tables

These are plain tables with JSON columns and no extensions, so any Postgres can hold them. The
gateway creates them on first connect.

- `mcp_calls`: `id`, `created` (timestamptz), `client`, `session`, `project`, `tool`,
  `arguments` (jsonb), `status` (`ok`/`error`), `seconds`, `summary`, `response` (jsonb).
- `mcp_call_refs`: `call_id`, `kind` (`run`, `run_id`, `request`, `doi`, `url`), `ref`. These rows
  are the backlinks.

Example queries:

```sql
-- What was searched, and when
SELECT created, tool, arguments->>'query' AS query, arguments->>'question' AS question
FROM mcp_calls ORDER BY created DESC LIMIT 50;

-- Every call that mentions a paper
SELECT c.created, c.tool, c.summary FROM mcp_calls c
JOIN mcp_call_refs r ON r.call_id = c.id
WHERE r.kind = 'doi' AND r.ref = '10.1121/10.0015244' ORDER BY c.created;
```

## Setting up Neon (once)

Neon's free plan is enough for this: 1 GB per project as of November 2026. Check
[neon.com/pricing](https://neon.com/pricing) for current limits.

1. **Create the project.** Sign up at [neon.com](https://neon.com) and create a project named
   `research-rag`. Pick the region nearest you (for example AWS us-west-2) and the newest Postgres
   version.
2. **Copy the connection string.** On the project dashboard, choose **Connect**, keep the default
   database (`neondb`), and copy the string, which looks like
   `postgresql://neondb_owner:…@ep-….us-west-2.aws.neon.tech/neondb?sslmode=require`. Either the
   pooled or the direct string works for the gateway.
3. **Store it in the Keychain, not in a file.** Run this, then paste the string when asked:

   ```bash
   security add-generic-password -a "$USER" -s com.polarizetech.research-rag.calllog -w
   ```

4. **Install the database driver** in the checkout the gateway runs from:

   ```bash
   cd ~/Sites/retrieval-augmented-generation && uv sync --extra calllog
   ```

5. **Give it to the gateway.** Add these lines to
   `~/Library/Application Support/PolarizeTech/research-rag/start.sh`, before the `exec` line:

   ```bash
   export CALL_LOG_DATABASE_URL="$(security find-generic-password -a "$USER" -s com.polarizetech.research-rag.calllog -w)"
   export RECORDS_URL="https://joshuas-macbook-pro.tailc463ae.ts.net:8892"
   ```

6. **Make the record pages reachable from your other devices, on the tailnet only.** Run this on
   the Mac's main Tailscale, not the `polarize-mcp` one:

   ```bash
   tailscale serve --bg --https=8892 http://localhost:8891
   ```

7. **Restart the gateway:**

   ```bash
   launchctl kickstart -k gui/$(id -u)/com.polarizetech.research-rag
   ```

8. **Check it.** `gateway_status` should show `"database": "configured"` and `"unsynced": 0`
   after a call or two. Then open
   `https://joshuas-macbook-pro.tailc463ae.ts.net:8892/records` from your phone.

The record pages, and their links, only open while the Mac is awake and on the tailnet. The
records themselves are always in Neon.

## Moving to DigitalOcean later

*Written in November 2026, before the move. Check the version numbers and DigitalOcean's
screens when you do it, and update this section afterwards.*

The database is ordinary Postgres, so the move is one export and one import. The gateway keeps
recording during the move: calls wait in the local outbox and are copied once it points at the
new database.

1. **Create the database.** In DigitalOcean, go to **Databases → Create → PostgreSQL** and pick a
   version the same as Neon's or newer. The smallest plan was about $15 a month in 2026. Add your
   Mac's public IP to **Trusted sources**, or the connection is refused. Copy the connection
   string from **Connection details** (it uses `sslmode=require` on port 25060).
2. **Install the Postgres client tools** to match the newer of the two server versions, for
   example `brew install libpq` and then put `$(brew --prefix libpq)/bin` on your `PATH`.
3. **Export from Neon.** Use Neon's *direct* connection string, the one without `-pooler` in the
   host name; `pg_dump` doesn't work through the pooler.

   ```bash
   pg_dump --format=custom --no-owner --no-acl \
     --table=mcp_calls --table=mcp_call_refs \
     "postgresql://…neon direct string…" > calllog.dump
   ```

4. **Import into DigitalOcean:**

   ```bash
   pg_restore --no-owner --no-acl --dbname="postgresql://…digitalocean string…" calllog.dump
   ```

5. **Compare the counts.** Run `SELECT count(*) FROM mcp_calls;` against both databases. A few
   more rows on Neon is fine: those calls are still in the local outbox and get copied in step 7.
6. **Swap the Keychain entry:**

   ```bash
   security add-generic-password -U -a "$USER" -s com.polarizetech.research-rag.calllog -w
   ```

   (`-U` updates the existing item; paste the DigitalOcean string.)
7. **Restart the gateway** with the `launchctl kickstart` command above. Rows made after the dump
   are copied from the outbox as they're found. Rows already copied to Neon after the dump are
   not, so make the dump while no research is running, or repeat steps 3–4 with
   `pg_restore --data-only`. Rows that are already there are skipped, because the tables use
   primary keys and the gateway inserts with `ON CONFLICT DO NOTHING`.
8. **Keep Neon for a week as a fallback,** then delete the project.

If the logs ever grow too big for a single dump, DigitalOcean's **Migrate a database** feature can
copy a live database instead.

## Settings

| Variable | Default | Meaning |
|---|---|---|
| `CALL_LOG_DATABASE_URL` | unset | Postgres connection string. Unset means the log stays local and nothing is pruned. |
| `CALL_LOG_LOCAL` | `<data dir>/calls.sqlite` | The local outbox file. |
| `CALL_LOG_LOCAL_DAYS` | `7` | Days a copied call stays in the outbox. |
| `RECORDS_URL` | `http://127.0.0.1:<port>` | Base address used in the links in replies. |
