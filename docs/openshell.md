# OpenShell: sandboxed agents on this Mac

[OpenShell](https://github.com/NVIDIA/OpenShell) is NVIDIA's open-source sandbox runtime for AI
agents (Apache 2.0). Each agent runs in its own sandbox. A policy decides which files it may read
or write, which hosts and programs it may reach, and which credentials it may use. Credentials
stay outside the sandbox: the agent sees a placeholder, and the real value is swapped in only on
requests to approved endpoints.

## Hard requirement: telemetry is OFF

OpenShell sends anonymous usage telemetry to NVIDIA by default, to
`https://events.telemetry.data.nvidia.com`. **Every OpenShell install used with this project
must have telemetry off, from the gateway's first start.** Never enable it, and never start a
gateway without the settings below.

| Control | Where | Value |
|---|---|---|
| Runtime switch | `~/.config/openshell/gateway.env`, also exported by the service wrapper | `OPENSHELL_TELEMETRY_ENABLED=false` |
| Second guard | `~/.config/openshell/gateway.env` | `OPENSHELL_TELEMETRY_ENDPOINT=http://127.0.0.1:9/telemetry-disabled` (a dead local port) |
| Sandboxes | propagated by the gateway into each sandbox supervisor | `OPENSHELL_TELEMETRY_ENABLED=false` |
| Helm (Kubernetes), if ever used | values | `server.telemetryEnabled=false` |
| Strongest option | build from source | `--no-default-features --features defaults-without-telemetry` for the gateway, sandbox and VM driver. This compiles the code out entirely; it needs a Rust toolchain. |

`scripts/openshell-telemetry-check.sh` checks all of this. It fails if:
- the env file is missing either setting;
- the running gateway's environment lacks `OPENSHELL_TELEMETRY_ENABLED=false`;
- any OpenShell process has a connection open to NVIDIA's telemetry host.

Run it after installing, after upgrading, and whenever the gateway is restarted.

What the release binaries contain (v0.1.8, checked with `strings`):
- **`openshell-gateway`** has the telemetry endpoint built in.
- **`openshell-driver-vm`** passes the switch into sandboxes.
- **`openshell`** (the CLI) and **`openshell-prover`** contain no telemetry code.

Source: the project's
[telemetry page](https://github.com/NVIDIA/OpenShell/blob/v0.1.8/docs/observability/telemetry.mdx)
and `crates/openshell-core/src/telemetry.rs`.

## Install on macOS without Homebrew

The official installer needs Homebrew and starts the gateway as a Homebrew service immediately.
This Mac has no Homebrew, so the formula's steps are done by hand. That also guarantees
telemetry is off before the first start.

1. Download the four `aarch64-apple-darwin` release assets (CLI, gateway, VM driver, prover) and
   the `*checksums-sha256.txt` files with `gh release download`. Check every archive with
   `shasum -a 256`.
2. Install:
   - `openshell`, `openshell-gateway` and `openshell-prover` into `~/.local/bin`;
   - `openshell-driver-vm` into `~/.local/libexec/openshell`, the gateway's first search
     directory.
3. Re-sign the VM driver with the hypervisor entitlement, as the formula does:
   `codesign --entitlements <plist with com.apple.security.hypervisor> --force -s - openshell-driver-vm`.
4. Write `~/.config/openshell/gateway.env` (the telemetry settings above, chmod 600) and
   `~/.config/openshell/gateway.toml`:

   ```toml
   [openshell]
   version = 2

   [openshell.gateway]
   compute_driver = "vm"
   ```

5. Generate the local mTLS material:
   `openshell-gateway generate-certs --output-dir ~/.local/state/openshell/tls --server-san host.openshell.internal`.
6. The VM driver needs `mke2fs` (e2fsprogs 1.43 or newer) to build sandbox disk images. Homebrew
   would provide it. Without Homebrew, build it from the kernel.org source tarball after checking
   its `sha256sums.asc`, and put it on the service wrapper's `PATH`.
7. Run `openshell-gateway config preflight --path ~/.config/openshell/gateway.toml` with the
   telemetry env loaded.
8. Start the gateway from `~/.local/libexec/openshell/openshell-gateway-service`, a wrapper that
   loads `gateway.env` and exports `OPENSHELL_TELEMETRY_ENABLED=false` before `exec`. Run it under
   a LaunchAgent, then register the gateway with
   `openshell gateway add https://127.0.0.1:17670 --local --name openshell`.
9. Run `scripts/openshell-telemetry-check.sh`.

The gateway listens on loopback only (127.0.0.1:17670).

## Reaching this project from a sandbox

From inside a sandbox the Mac is `host.openshell.internal`, never `127.0.0.1` or `localhost`.
The research tool is `http://host.openshell.internal:8891/`, and it needs a network policy entry
for the agent's binary. Policies are default-deny: a path or host not listed is blocked.

## Caveats

- OpenShell is young: v0.1.8, moving to the CNCF. Re-read the release notes before upgrading.
- The release binaries are ad-hoc signed, with no developer certificate. The checksums come from
  the same release page, so they guard against corruption, not against a compromised release.
- OpenShell limits what a compromised agent can reach. It doesn't stop prompt injection, and it
  doesn't replace least-privilege tokens or review.
