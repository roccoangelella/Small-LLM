# Operational Decisions

_Last updated: 2026-08-02_

## 2026-08-02 — Launch the authenticated 10M-token dataset pilot

The user authorized running the bounded 10M-token dataset acceptance pilot on the VPS through PiLink.

Execution contract:

- the VPS checkout must be clean and exactly match the repository `main` branch before launch;
- use the approved exact ClimbMix weight artifact with SHA-256 `76e82e22760adcac59c7294fe9bac11358f5a8b7a26035aae64c3f2e6fa1acb7`;
- use the committed authenticated pilot bounds: 10,000,000 target accepted source tokens, 9,000,000 minimum, 11,000,000 hard maximum, and 2,000,000-token durable checkpoint cadence;
- require the real personal-Google-Drive mirror and authorized-user OAuth credentials;
- do not substitute `--allow-local-only` or a synthetic smoke run for the authenticated acceptance pilot;
- retain durable logs, exit codes, interruption snapshot, completed-resume baseline, manifests, hashes, and acceptance reports under the documented `/data/climbmix-*` paths;
- complete the intentional interruption/resume, schema-v2 full verification, completed-resume idempotence, and fail-closed acceptance verification before declaring the pilot passed.

Launch is permitted only after the current checkout's operational prerequisites are present and validated, including the approved calibration artifact, sufficient disk, remote dependencies, passing offline evidence, OAuth token, and Drive folder configuration.

## 2026-08-02 — Authenticated 10M-token dataset pilot passed

The authorized pilot completed successfully at repository commit `e4776501d68e39746f8a75dcbb9c49515f215abd`. The accepted run used the approved exact mixture file, real personal-Google-Drive mirroring, and the committed 10M/9M/11M bounds with a 2M-token checkpoint cadence.

The accepted evidence records:

- a durable incomplete snapshot at 2,000,112 accepted source tokens and 2,814 documents;
- termination of the actual producer process group with exit status 143;
- successful continuation with the identical production command plus `--resume`;
- final completion at 10,000,662 accepted source tokens and 14,136 documents;
- seven local immutable shards and seven matching Drive entries;
- successful full schema-v2 verification;
- successful completed-resume idempotence;
- fail-closed acceptance status `PASS` for environment, calibration, offline tests, calibration run, Drive smoke, pilot, interruption/resume, and completed-resume idempotence.

The canonical acceptance report is `/data/climbmix-ops/dataset_acceptance_report.json`, SHA-256 `b18decde4aa0e6e7376c3fecd3dda4406dee983f11224537cf73dd22a66bc00b`.

A prior attempt that did not terminate the actual producer was explicitly rejected, archived, and excluded from the accepted evidence. This prevents a wrapper-level signal from being misrepresented as an interruption/resume qualification.
