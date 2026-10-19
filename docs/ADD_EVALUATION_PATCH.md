# Additive evaluation patch

Copy the included new files into the existing UrbanFlow project root, preserving the folder structure. These files are additive and should not overwrite existing source files:

- `scripts/evaluate_multihorizon_verified.py`
- `notebooks/12_verified_evaluation.ipynb`
- `docs/VERIFIED_EVALUATION_GUIDE.md`

Then run from the project root:

```bash
./.venv/bin/python scripts/evaluate_multihorizon_verified.py
```

Do not run any target-generation cell to create missing data before checking the audit. The script requires all four test target arrays and the raw METR-LA HDF5 file; if they are not present, it stops and names the missing file(s). It never retrains the model and never overwrites historical results or target/mask files.
