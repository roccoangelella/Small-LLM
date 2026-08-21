# R-SFT step-417 final-upload SIGKILL — 2026-08-21

The expanded-corpus production run `100m-2b-rsft-r0-16716-001` reached the final optimizer boundary at step 417. During publication, Hugging Face began uploading `step-00000417/last/trainer_state.pkl`, whose checkpoint manifest records 913,854,883 bytes and SHA-256 `6adb4f520a63c971dd983a878996cacd692a5c10f66dff518d33db64955ae881`.

The notebook log showed the Xet upload at approximately 211 MB before a long stall with repeated `IOStream.flush timed out` messages. Rank zero was then terminated by SIGKILL; rank one subsequently failed its Gloo control barrier with `Connection closed by peer` and was terminated by the elastic launcher.

Remote inspection after failure found the step-417 `checkpoint.json`, `drive_manifest.json`, `local_manifest.json`, and `checkpoint_manifest.json`, but no step-417 `trainer_state.pkl`. Crucially, `run/100m-2b-rsft-r0-16716-001/latest.json` still points to verified `step-00000250`, whose complete trainer state remains present. The two-phase publication gate therefore failed closed as designed.

The R-SFT launcher now sets `HF_HUB_DISABLE_XET=1` and `HF_HUB_DISABLE_PROGRESS_BARS=1` for the DDP process. With `huggingface_hub==1.5.0`, a direct runtime probe confirmed `HF_HUB_DISABLE_XET=True` and progress bars disabled. The next launch should use the same run ID so exact-resume starts from step 250 and replays the final 167 optimizer steps.
