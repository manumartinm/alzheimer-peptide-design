# Peptide Folding via the Boltz API

> Status: IMPLEMENTED (`bbb_dataset.cli.fold` / `uv run bbb-dataset-fold`). Requires Boltz API credentials.

## 1. Goal

Fold every unique peptide sequence in the gold dataset (`packages/dataset/data/processed/peptides_bbb.parquet`) into a 3D structure, once, offline, and store coordinates + confidence in a structural manifest. These structures are static training data; folding is not done at BoltzGen inference time (BoltzGen already produces coordinates there).

## 2. Why the hosted Boltz API

We use the hosted Boltz API (https://api.boltz.bio/docs/) instead of a local Boltz install:

- no local GPU/model setup,
- consistent, versioned model (e.g. `boltz-2.1`),
- simple batch submission for ~450 peptides.

Trade-offs: requires `BOLTZ_API_KEY`, consumes credits, and is subject to rate limits and network failures. The folding script must therefore be asynchronous, cached, and resumable.

## 3. Authentication

Add to `.env` / `.env.example`:

```
# Boltz API key (https://api.boltz.bio)
BOLTZ_API_KEY=
```

The script reads `BOLTZ_API_KEY` from the environment. Never commit the real key.

## 4. Job definition

Each peptide is a single-entity `protein` structure prediction (`predictions:structure-and-binding`). Conceptual input:

```yaml
entities:
  - type: protein
    value: <PEPTIDE_SEQUENCE>
    chain_ids: ["A"]
```

Preferred driver: the Python SDK (`run()` downloads to `boltz-experiments/<name>/`). Credentials live in `dataset/.env.local`.

Per-job outputs (see [Predict structure and binding](https://api.boltz.bio/docs/guides/predictions/)):

```
boltz-experiments/<name>/
  run.json                  # prediction object with metrics
  outputs/files/
    sample_0.cif            # best / first sample structure
```

Metrics are read from `run.json` → `output.best_sample.metrics` (or the first entry in `all_sample_results`).

## 5. CLI: `bbb-dataset-fold`

Location: `packages/dataset/src/bbb_dataset/cli/fold.py` (folding belongs to dataset construction).

Run from the dataset root:

```bash
cd packages/dataset
uv run bbb-dataset-fold
```

Responsibilities:

1. Read the gold parquet; collect unique sequences.
2. For each sequence, call the Boltz API (`run()`).
3. Parse metrics from `run.json` and backbone coords from the downloaded CIF.
4. Save `coords.npz` and write the manifest parquet.

Output layout: `data/structures/<sequence_hash>/coords.npz` (no resume cache; re-run overwrites).

## 6. Manifest schema

`data/processed/peptides_struct_manifest.parquet`:

| Column | Description |
|--------|-------------|
| `peptide_id` | Stable id from the dataset pipeline |
| `sequence_hash` | SHA256 prefix of the sequence (output folder name) |
| `coords_path` | Path to the npz with backbone/Ca coordinates |
| `plddt` | `complex_plddt × 100` from Boltz (0–100 scale for training weights) |
| `ptm` | Global predicted TM-score (0–1) |
| `structure_confidence` | Overall structure confidence (0–1) |
| `iptm` | Interface predicted TM-score (0–1) |
| `complex_iplddt` | Interface pLDDT (0–1) |
| `complex_pde` | Predicted distance error (Å, lower is better) |
| `complex_ipde` | Interface predicted distance error (Å) |
| `length` | Peptide length |

## 7. Confidence handling

Short, flexible peptides (5-30 aa) often fold with low confidence. Downstream:

- training samples are weighted by `plddt` (low-confidence structures contribute less);
- noise-aware training further reduces over-reliance on any single static conformation;
- optionally drop peptides below a `plddt` floor (configurable).

## 8. Operational notes

- Run once; treat the manifest as a cached, versioned artifact (track with DVC if desired).
- Respect API rate limits with a bounded concurrency semaphore and exponential backoff on transient errors.
- The script must be safely re-runnable: completed hashes are skipped, only missing/failed ones are retried.
