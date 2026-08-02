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
