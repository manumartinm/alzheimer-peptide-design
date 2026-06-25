# Dataset data directory

Large data files are not committed to git. Regenerate locally or download from Hugging Face.

## Regenerate

```bash
cd packages/dataset
uv run tfg-bbb-build
uv run tfg-bbb-export-hf --variant full   # writes data/hf_release/
```

## Hugging Face

Pre-built release: [`manumartinm/bbb-peptides`](https://huggingface.co/datasets/manumartinm/bbb-peptides) (825 peptides with Boltz structures).

## Layout

| Path | Description |
|------|-------------|
| `raw/` | Source FASTA/TSV (gitignored, downloaded by build) |
| `processed/` | Gold parquet tables + EDA summaries |
| `hf_release/` | HF export bundle (gitignored) |
| `structures/` | Full structure store (gitignored) |
