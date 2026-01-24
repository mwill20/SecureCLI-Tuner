# Data Provenance & Methodology [DRAFT]

We are currently verifying the dataset splits and provenance metadata to ensure they align exactly with the 1,227 examples used in the final training run.

**Verification Checklist:**

- [ ] SHA256 Hash of `train.jsonl`
- [ ] Correct Test Set distribution (currently correcting discrepancy between 125 vs 1227 reports).
- [ ] Re-validating cleaning logic in `preprocess_data.py`.
