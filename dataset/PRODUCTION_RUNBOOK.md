# Dataset Production Runbook

This runbook covers the dataset-only production path. It builds the schema-v2
streaming cache, enforces the frozen corpus-size envelope, and mirrors every
committed immutable shard to Google Drive before advancing the durable source
cursor.

The trainer/model checkpoint path remains in `dataset.src.joint_checkpoint` and
is intentionally not started by this command.

> [!WARNING]
> **Authentication Requirement**: Personal Google Drive storage uses installed-app OAuth credentials (`.secrets/google-drive-authorized-user.json`). Service accounts and API keys are **not** supported and will be rejected automatically.

## Production gates

Do not start the 90B-token run until all of the following are true:

1. The cluster-weight JSON has been reviewed and approved. The repository does
   not provide a production default.
2. Personal Google Drive OAuth setup has completed cleanly via `dataset.drive_auth setup`.
3. The bounded authenticated pilot below passes, including interruption and
   resume.
4. `python -m unittest discover -v` and the GitHub Actions workflow are green.
5. The output volume has at least the preflight-required free space. The command
   sizes this requirement against the 100B hard maximum by default.

## Environment & OAuth Setup

Use Python 3.13 and install the optional remote dependencies:

```bash
uv sync --locked
uv pip install -r dataset/requirements-remote.txt
```

### 1. Google Cloud Console Setup

1. Open [Google Cloud Console](https://console.cloud.google.com/).
2. Create or select a Google Cloud project.
3. Configure the **OAuth consent screen**:
   - User Type: External (or Internal if using Google Workspace).
   - App name: `Small LLM Storage`.
   - Scope: `https://www.googleapis.com/auth/drive.file`.
   - Add your personal Google email as a test user if the app is in testing mode.
4. Create Credentials:
   - Click **Create Credentials** -> **OAuth client ID**.
   - Application type: **Desktop app**.
   - Download the JSON credentials file and place it at `.secrets/google-drive-oauth-client.json`.

### 2. Run the Interactive OAuth Setup

Run the setup command:

```bash
uv run python -m dataset.drive_auth setup \
  --client-secrets .secrets/google-drive-oauth-client.json \
  --token-file .secrets/google-drive-authorized-user.json
```

What this setup command does:

1. **Validates** that `.secrets/google-drive-oauth-client.json` is a Desktop App client secrets JSON (rejecting service-account files and web client secrets).
2. **Requests scope** `https://www.googleapis.com/auth/drive.file` and opens your browser for one-time Google account consent with offline refresh access.
3. **Atomically writes** `.secrets/google-drive-authorized-user.json`. Subsequent runs automatically load and refresh this token without opening the browser.
4. **Creates or reuses** the application folder tree in your personal My Drive:
   ```text
   Small LLM Storage/
   └── dataset-shards/
   ```
   Folder lookups check existing directories first to avoid duplicate folders.
5. **Saves environment variables** to `.env` (gitignored):
   ```env
   SMALL_LLM_GOOGLE_OAUTH_TOKEN=.secrets/google-drive-authorized-user.json
   SMALL_LLM_DRIVE_FOLDER_ID=<folder-id>
   ```
6. **Performs an automated smoke test**: uploads a temporary test payload to `dataset-shards`, reads object metadata, downloads and checks SHA-256 and MD5 checksums, and cleans up the test object.
7. **Reports** the authenticated Google account email and folder ID without printing secrets or access tokens.

### 3. Credentials & Security Rules

- Never commit `.env` or `.secrets/` files. Both are covered in `.gitignore`.
- Production and acceptance CLIs automatically read `SMALL_LLM_GOOGLE_OAUTH_TOKEN` and `SMALL_LLM_DRIVE_FOLDER_ID` from the environment or `.env`.
- CLI overrides `--google-oauth-token` (or `--google-credentials`) and `--drive-folder-id` take precedence.

### 4. Revoking Authorization

If you ever need to revoke access:
- Go to [Google Account Permissions](https://myaccount.google.com/permissions).
- Select `Small LLM Storage` and click **Remove Access**.
- Delete `.secrets/google-drive-authorized-user.json` locally and re-run `dataset.drive_auth setup`.

## Validate the approved weights

The existing offline preflight remains useful:

```bash
uv run python -m dataset.main stream-cache \
  --weights-file /secure/approved-cluster-weights.json \
  --show-stream-config
```

Record the exact file checksum in the operational change ticket:

```bash
sha256sum /secure/approved-cluster-weights.json
```

The production manifest additionally freezes the normalized stream configuration,
work-plan hash, schema hash, and complete production policy in one configuration
hash. Resume refuses any mismatch.

## Authenticated bounded pilot

The pilot uses the real pinned source and real Drive folder, but stops after a
small whole-document source-token target:

```bash
uv run python -m dataset.production \
  --weights-file /secure/approved-cluster-weights.json \
  --output-dir /data/climbmix-pilot \
  --run-id climbmix-pilot-001 \
  --target-tokens 10000000 \
  --minimum-tokens 9000000 \
  --maximum-tokens 11000000 \
  --checkpoint-source-tokens 2000000 \
  --allow-unsafe-low-disk
```

The local-only escape hatch is restricted to synthetic/development checks:

```bash
uv run python -m dataset.production \
  --weights-file /secure/approved-cluster-weights.json \
  --output-dir /tmp/climbmix-local-smoke \
  --run-id local-smoke \
  --target-tokens 100000 \
  --minimum-tokens 90000 \
  --maximum-tokens 110000 \
  --checkpoint-source-tokens 25000 \
  --allow-local-only \
  --allow-unsafe-low-disk
```

Never use `--allow-local-only` for the production corpus.

## Interruption and resume acceptance test

During the authenticated pilot, terminate the process after at least one durable
checkpoint. Resume with byte-for-byte identical arguments plus `--resume`:

```bash
uv run python -m dataset.production \
  --weights-file /secure/approved-cluster-weights.json \
  --output-dir /data/climbmix-pilot \
  --run-id climbmix-pilot-001 \
  --target-tokens 10000000 \
  --minimum-tokens 9000000 \
  --maximum-tokens 11000000 \
  --checkpoint-source-tokens 2000000 \
  --allow-unsafe-low-disk \
  --resume
```

Resume replays the immutable work plan to the committed source cursor, removes
uncommitted local shard tails or orphan finalized files, verifies previously
committed Drive objects once, and uploads any missing deterministic shards.

A finalization backup protects the last remotely durable cursor from the
underlying producer's temporary `progress.json` rewrite. If the process or VPS
fails during final commit, the next `--resume` restores that backup before
rebuilding deterministic orphan shards.

## Verification

After the pilot completes:

```bash
uv run python -m dataset.main verify \
  --output-dir /data/climbmix-pilot \
  --full-scan
```

Inspect these files:

- `manifest.json`: schema-v2 local shard and production-policy manifest.
- `progress.json`: final committed producer/source cursor with `complete: true`.
- `drive_manifest.json`: only shards referenced by the completed local manifest,
  including Drive IDs, local SHA-256 values, and remote verification state.
- `work_plan.json`: pinned and self-hashed source-region order.

Acceptance requires:

- accepted source tokens are at least the minimum and never above the hard maximum;
- the completion reason is expected (`target_reached`, `source_exhausted`, or the
  rare whole-document `hard_maximum_guard`);
- every final shard has one `remote_durable: true` Drive-manifest entry;
- a second completed `--resume` performs verification without uploading duplicate
  objects;
- no `.tmp` shard or `progress.production.safe.json` remains.

## Full production run

After the pilot report and weights are approved:

```bash
uv run python -m dataset.production \
  --weights-file /secure/approved-cluster-weights.json \
  --output-dir /data/climbmix-cache \
  --run-id climbmix-production-v1
```

Defaults enforce:

- target: 90,000,000,000 accepted source tokens;
- minimum: 80,000,000,000;
- hard maximum: 100,000,000,000;
- durable checkpoint cadence: 1,000,000,000 accepted source tokens;
- remote durability required;
- 1 GiB target shards;
- the pinned Nemotron-ClimbMix revision and cluster-11 exclusion already frozen
  in `dataset.config`.

The 1B-token checkpoint cadence replaces the unsafe previous per-document
checkpoint default. It bounds restart loss while avoiding one immutable shard and
full filesystem synchronization cycle per document.

## Failure handling

- A transient HTTP/provider error is retried with bounded exponential backoff.
- A deterministic checksum, identity, configuration, or cursor mismatch aborts
  immediately and must not be bypassed.
- If Drive publication fails, the durable cursor is not advanced. Resume removes
  local artifacts newer than the last committed cursor and deterministically
  rebuilds them. Already-uploaded matching Drive objects are reused.
- The advisory run lock rejects concurrent writers targeting the same output
  directory.
- Remote objects left outside the final manifest are retained for forensic
  inspection; deletion remains an explicit, verification-gated operation.
