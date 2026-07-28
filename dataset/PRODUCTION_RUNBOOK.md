# Dataset Production Runbook

This runbook covers the dataset-only schema-v2 cache path. It streams the pinned Nemotron-ClimbMix release, enforces the corpus-size envelope, and mirrors every committed immutable shard to Google Drive before advancing the durable source cursor.

The trainer and model checkpoint path remains separate and is intentionally not started by these commands.

> [!WARNING]
> Personal Google Drive storage uses installed-app OAuth credentials. Service accounts and API keys are not supported for this project configuration.

## Production gates

Do not start the 90B-token run until all of the following are true:

1. The exact cluster-weight JSON has been generated, reviewed, and approved.
2. Personal Google Drive OAuth setup has passed its real upload/download smoke test.
3. The authenticated bounded 10M-token pilot has passed interruption, resume, verification, idempotence, and cleanup checks.
4. The repository unit tests and GitHub Actions workflow are green.
5. The output volume passes disk preflight for the intended run.
6. The trainer consumer and joint-checkpoint integration are ready for the production launch.

## Environment and dependencies

Use Python 3.13 and install the remote dependencies:

```bash
uv sync --locked
uv pip install -r dataset/requirements-remote.txt
```

The following paths are local secrets and must never be committed:

```text
.secrets/google-drive-oauth-client.json
.secrets/google-drive-authorized-user.json
.env
```

`.secrets/` and `.env` are covered by `.gitignore`.

## Personal Google Drive OAuth setup

The Google Cloud project must have:

- Google Drive API enabled;
- an OAuth consent screen;
- a Desktop App OAuth client;
- the account running the setup added as a test user while the app is in testing mode;
- scope `https://www.googleapis.com/auth/drive.file`.

Place the downloaded Desktop App client JSON at:

```text
.secrets/google-drive-oauth-client.json
```

Run:

```bash
uv run python -m dataset.drive_auth setup \
  --client-secrets .secrets/google-drive-oauth-client.json \
  --token-file .secrets/google-drive-authorized-user.json
```

The command:

1. validates the OAuth client type;
2. opens the browser for one-time authorization;
3. obtains and atomically stores an authorized-user refresh token;
4. reuses and refreshes that token on later runs;
5. creates or reuses:

```text
Small LLM Storage/
└── dataset-shards/
```

6. writes these values to `.env`:

```env
SMALL_LLM_GOOGLE_OAUTH_TOKEN=.secrets/google-drive-authorized-user.json
SMALL_LLM_DRIVE_FOLDER_ID=<dataset-shards-folder-id>
```

7. performs a real upload, metadata read, download checksum verification, and cleanup smoke test.

Successful completion must end with `Smoke test: PASSED`.

### Loading `.env`

`dataset.production` reads process environment variables or explicit CLI arguments. It does not currently parse `.env` itself.

Therefore run production commands with:

```bash
uv run --env-file .env python -m dataset.production ...
```

Alternatively pass both explicitly:

```text
--google-oauth-token .secrets/google-drive-authorized-user.json
--drive-folder-id <dataset-shards-folder-id>
```

Do not use `DRIVE_API_KEY`, `GOOGLE_APPLICATION_CREDENTIALS`, or a service-account file for the personal-Drive production path.

## Exact mixture calibration

The production weights must come from a complete scan of the pinned release:

```bash
uv run python -m dataset.mixture \
  --output-dir /data/climbmix-mixture-calibration \
  --workers 8 \
  --max-in-flight-work-items 16
```

Resume an interrupted scan with the same output directory:

```bash
uv run python -m dataset.mixture \
  --output-dir /data/climbmix-mixture-calibration \
  --workers 8 \
  --max-in-flight-work-items 16 \
  --resume
```

Expected outputs:

```text
work_plan.json
mixture_progress.json
mixture_report.json
climbmix_code_free_weights.json
```

Review the report and record the approved weight-file checksum:

```bash
sha256sum /data/climbmix-mixture-calibration/climbmix_code_free_weights.json
```

Do not use rounded paper percentages as production weights.

## Authenticated bounded pilot

Use the approved exact weight file and the real Drive folder:

```bash
uv run --env-file .env python -m dataset.production \
  --weights-file /data/climbmix-mixture-calibration/climbmix_code_free_weights.json \
  --output-dir /data/climbmix-pilot \
  --run-id climbmix-pilot-001 \
  --target-tokens 10000000 \
  --minimum-tokens 9000000 \
  --maximum-tokens 11000000 \
  --checkpoint-source-tokens 2000000 \
  --allow-unsafe-low-disk
```

`--allow-unsafe-low-disk` is acceptable only for the deliberately bounded pilot when the operator has independently confirmed sufficient space. Never use it casually for the full corpus.

The local-only escape hatch is restricted to synthetic development checks:

```bash
uv run python -m dataset.production \
  --weights-file /path/to/test-weights.json \
  --output-dir /tmp/climbmix-local-smoke \
  --run-id local-smoke \
  --target-tokens 100000 \
  --minimum-tokens 90000 \
  --maximum-tokens 110000 \
  --checkpoint-source-tokens 25000 \
  --allow-local-only \
  --allow-unsafe-low-disk
```

Never use `--allow-local-only` for production or the authenticated acceptance pilot.

## Interruption and resume test

After at least one durable checkpoint has been committed and its referenced shards are remotely durable, terminate the pilot process.

Resume with identical semantic arguments plus `--resume`:

```bash
uv run --env-file .env python -m dataset.production \
  --weights-file /data/climbmix-mixture-calibration/climbmix_code_free_weights.json \
  --output-dir /data/climbmix-pilot \
  --run-id climbmix-pilot-001 \
  --target-tokens 10000000 \
  --minimum-tokens 9000000 \
  --maximum-tokens 11000000 \
  --checkpoint-source-tokens 2000000 \
  --allow-unsafe-low-disk \
  --resume
```

Resume must:

- restore the immutable work plan and committed source cursor;
- reject configuration, schema, policy, weight, or source drift;
- remove uncommitted local tails and deterministic orphan files;
- reuse already-uploaded matching Drive objects;
- continue without silently skipping or double counting source ranges.

## Verification

After the pilot completes:

```bash
uv run python -m dataset.main verify \
  --output-dir /data/climbmix-pilot \
  --full-scan
```

Inspect:

- `manifest.json`;
- `progress.json`;
- `drive_manifest.json`;
- `work_plan.json`.

Acceptance requires:

- accepted source tokens are at least 9M and at most 11M;
- `progress.json` is complete with a valid completion reason;
- every finalized local shard matches its recorded size and SHA-256;
- every finalized shard has exactly one matching remotely durable Drive entry;
- no referenced Drive object is missing;
- no `.tmp`, `.part`, smoke-test, or `progress.production.safe.json` artifact remains.

## Completed-resume idempotence

Record the final local hashes and Drive file IDs, then run the same completed command again with `--resume`.

Acceptance requires:

- successful exit;
- unchanged accepted-source-token count;
- unchanged local shard hashes;
- unchanged Drive file IDs;
- no duplicate logical shard;
- no additional immutable shard upload;
- unchanged final source cursor.

## Full production launch

Do not launch this until the trainer integration and bounded end-to-end training pilot are ready.

At launch time, start the dataset producer first, establish a bounded cache head start, then start training while dataset preparation and Drive mirroring continue concurrently.

The production command is:

```bash
uv run --env-file .env python -m dataset.production \
  --weights-file /data/climbmix-mixture-calibration/climbmix_code_free_weights.json \
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
- pinned Nemotron-ClimbMix revision;
- cluster-11 exclusion.

## Failure handling

- Transient provider errors use bounded retry behavior.
- Deterministic checksum, identity, configuration, source, or cursor mismatches abort immediately.
- If Drive publication fails, the durable cursor is not advanced.
- Resume removes local artifacts newer than the last committed cursor and deterministically rebuilds them.
- Already-uploaded matching Drive objects are reused.
- The advisory run lock rejects concurrent writers targeting the same output directory.
- Remote objects outside the final manifest are retained for explicit forensic inspection and cleanup.
