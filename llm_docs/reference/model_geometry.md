# Model geometry

_Last reviewed: 2026-08-13_

The model family is geometry-scalable; parameter labels are approximate names, while the implemented parameter counter is authoritative.

## Completed hybrid geometries

| quantity | 20M | 100M |
|---|---:|---:|
| exact learned parameters | 20,637,592 | 101,252,280 |
| context | 2,048 | 2,048 |
| `d_model` | 256 | 512 |
| decoder layers | 8 | 20 |
| GDN-2 layers | 6 | 15 |
| gated full-MHA layers | 2 | 5 |
| `d_ff` | 704 | 1,408 |
| attention heads | 4 | 8 |
| head dimension | 64 | 64 |
| GDN key/value heads | 4 | 8 |
| GDN key/value dimension per head | 64 | 64 |
| GDN short-conv kernel | 4 | 4 |
| serialized `gdn_chunk_size` | 32 | 32 |
| tied embeddings | yes | yes |
| semantic / padded vocab | 50,257 / 50,304 | 50,257 / 50,304 |

FLA's production internal runtime chunk is 64 and is **not** a learned/model-geometry parameter. Do not rewrite saved checkpoint geometry from 32 to 64 merely because the selected kernel executes 64-token internal chunks.

## 100M parameter breakdown

The implemented approximately-100M hybrid has:

| component | parameters |
|---|---:|
| tied embedding/output | 25,755,648 |
| 20 SwiGLU FFNs | 43,253,760 |
| 5 gated MHA mixers | 6,554,240 |
| 15 GDN-2 mixers | 25,667,640 |
| remaining block/final RMSNorms | 20,992 |
| **total** | **101,252,280** |

The approximately-20M hybrid total is **20,637,592**.

## Parameter-matched transformer references

For the 100M geometry, replacing the 15 GDN-2 mixers with gated attention uses compensating `d_ff=1603`, yielding 101,237,760 parameters for either the SWA/full-MHA pattern or all-full-MHA schedule. The small difference from the hybrid is due to integral FFN width. These are comparison geometries, not completed primary production runs.

## Planning templates

Unrun larger geometries are planning templates only and are not authorization:

| role | approximate params | d_model | layers | hybrid d_ff | heads × dimension |
|---|---:|---:|---:|---:|---:|
| intermediate debug | 44M | 384 | 12 | 1,024 | 6 × 64 |
| medium trial | ~200M | 768 | 20 | 2,048 | 12 × 64 |
| serious trial | ~344M | 1,024 | 20 | 2,816 | 16 × 64 |

A near-1B model remains a long-term project goal. Any next geometry requires a measured scaling decision; do not infer authorization from this table.

## Hardware/accounting rule

Chunk size, microbatch, DDP world size, and backend selection affect execution/memory/throughput but do not change learned parameter count. Exact checkpoint/model config and the implemented counter take precedence over approximate model labels.
