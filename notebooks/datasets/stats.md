# Scorio dataset statistics

## Release summary

| dataset | configs | tasks | models | questions | logical attempts | physical rows | Parquet files | Parquet size |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| [Scorio GPQA](https://huggingface.co/buckets/harimo/scorio-gpqa) | 4 | 1 | 4 | 3,600 | 1,152,000 | 1,152,000 | 14,400 | 679.38 GiB |
| [Scorio Lite](https://huggingface.co/datasets/harimo/scorio-lite) | 10 | 6 | 4 | 3,786 | 1,211,520 | 2,423,040 | 328 | 35.22 GiB |
| [Scorio Math](https://huggingface.co/buckets/harimo/scorio-math) | 4 | 5 | 4 | 186 | 59,520 | 59,520 | 744 | 166.77 GiB |
| [Scorio Trace](https://huggingface.co/datasets/harimo/scorio-trace) | 21 | 4 | 20 | 120 | 192,000 | 384,000 | 160 | 13.08 GiB |

The four releases contain 15,632 Parquet files, 4,018,560 physical rows, and
894.45 GiB of data. After accounting for copies shared across releases and
tiers, there are 1,403,520 unique sampled attempts: 1,211,520 from the Math and
GPQA collections, plus 192,000 from Scorio Trace.

### Counts

- `configs` counts loadable dataset or bucket configurations.
- Each question has 80 sampled attempts per model.
- `logical attempts` counts sampled model responses. Copies do not add to this
  number.
- `physical rows` counts every stored Parquet row, including full and
  metadata-only copies.
- File sizes cover Parquet data only. GiB values use binary units and are
  rounded to two decimal places.

Scorio Lite contains compact copies of the Scorio GPQA and Scorio Math attempts.
Its `meta` tiers contain those rows again, without token-level lists. Scorio
Trace also stores each attempt twice: once in a per-model full tier and once in
the combined `meta` tier. For that reason, adding the four `logical attempts`
cells would double-count data.

## Physical layout

| dataset | tier | family | files | rows | 80-row pools | Parquet size |
|---|---|---|---:|---:|---:|---:|
| Scorio GPQA | full | GPQA | 14,400 | 1,152,000 | 14,400 | 679.38 GiB |
| Scorio Lite | full | GPQA | 144 | 1,152,000 | 14,400 | 27.04 GiB |
| Scorio Lite | full | math | 20 | 59,520 | 744 | 6.28 GiB |
| Scorio Lite | meta | GPQA | 144 | 1,152,000 | 14,400 | 1.45 GiB |
| Scorio Lite | meta | math | 20 | 59,520 | 744 | 0.44 GiB |
| Scorio Math | full | math | 744 | 59,520 | 744 | 166.77 GiB |
| Scorio Trace | full | math | 80 | 192,000 | 2,400 | 11.72 GiB |
| Scorio Trace | meta | math | 80 | 192,000 | 2,400 | 1.36 GiB |

In Scorio GPQA and Scorio Math, each file contains one 80-attempt candidate
pool, although a large file may contain several internal row groups. In Scorio
Lite and Scorio Trace, each Parquet row group contains one candidate pool.

## Release checks

All 44 release checks passed. The checks covered:

- published file paths, byte sizes, and row counts against the release manifests;
- Parquet readability;
- the expected 80 seeds in each candidate pool, in order;
- candidate-pool boundaries at the file or row-group level;
- configuration, task, model, and question counts; and
- zero recorded build failures.

## CSV data

- [tables/release_summary.csv](tables/release_summary.csv): one row per published dataset
- [tables/release_tiers.csv](tables/release_tiers.csv): full/meta and math/GPQA storage breakdown
- [tables/release_integrity.csv](tables/release_integrity.csv): every release-level check

Schemas, loading instructions, and example notebooks are listed in the
[dataset guide](README.md).
