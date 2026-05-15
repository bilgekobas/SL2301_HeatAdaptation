# Cleaned 03 analysis scripts

These scripts use repository-relative paths and expect this folder structure:

```text
<repo>/
├── code/03_analysis/
├── data/processed/
├── data/metadata/
└── outputs/03_analysis/
```

Expected metadata files:

- `data/metadata/participant_meta_pseud.csv`
- `data/metadata/session_meta.csv`

Expected processed files currently referenced by the scripts:

- `data/processed/bp.csv`
- `data/processed/thermal_comfort.csv`
- `data/processed/skin_temp.csv`
- `data/processed/cbt.csv`
- `data/processed/hrv.csv`

Run scripts from the repository root or directly from `code/03_analysis`; paths are resolved using `__file__`, not the working directory.
