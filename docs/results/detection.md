# Detection results

**Pending: no real-data results exist yet.** FaceForensics++ and Celeb-DF have not been downloaded,
so no video has been extracted and no detector has been trained on real footage. There are no
numbers here. Running the last command below replaces this file with the generated tables
(nothing is entered by hand).

The only detector numbers in the repository are from a **synthetic** showcase on simulated faces,
described and caveated in [../METHOD.md](../METHOD.md) section 19. They say nothing about real
deepfakes.

## How to produce the results

Run from the project folder in PowerShell, after downloading FaceForensics++ (videos only, c23 and
c40) and Celeb-DF v2 into the folders below (details in the [README](../../README.md)):

```powershell
$env:PULSECHECK_DATA = "C:\dev\pulsecheck-data"

# 1. official FaceForensics++ split files
C:\dev\venvs\pulsecheck\Scripts\python scripts/fetch_ffpp_splits.py

# 2. one manifest per dataset (label, leakage group, split, compression)
C:\dev\venvs\pulsecheck\Scripts\python scripts/build_manifest.py `
    --ffpp-root C:\dev\pulsecheck-data\ffpp `
    --celebdf-root C:\dev\pulsecheck-data\celebdf `
    --compression c23 c40 --max-train-pairs 200

# 3. face tracking and colour traces: the slow step, run it overnight; re-run the same command to resume
C:\dev\venvs\pulsecheck\Scripts\python scripts/extract_all.py `
    --manifest C:\dev\pulsecheck-data\manifests\ffpp_c23.csv `
    --manifest C:\dev\pulsecheck-data\manifests\ffpp_c40.csv `
    --manifest C:\dev\pulsecheck-data\manifests\celebdf_test.csv `
    --workers 4

# 4. one feature table per manifest (reads cached traces only)
foreach ($m in "ffpp_c23", "ffpp_c40", "celebdf_test") {
    C:\dev\venvs\pulsecheck\Scripts\python scripts/build_features.py `
        --manifest C:\dev\pulsecheck-data\manifests\$m.csv
}

# 5. train and evaluate; overwrites this file and writes docs/figures/roc_e2.png and roc_e4.png
C:\dev\venvs\pulsecheck\Scripts\python scripts/run_experiments.py `
    --features-dir C:\dev\pulsecheck-data\features
```

## What will be reported

The generated file will contain the protocol, the number of real and fake videos per table and split,
the share of videos for which each feature could not be computed, and:

- **E2:** FF++ c23, trained on the official train split, threshold on val, scored on test. Feature
  groups A, AB, ABC and ABCD, with logistic regression and gradient boosting: validation and test
  ROC-AUC, balanced accuracy, precision and recall at the validation Youden threshold, 5-fold
  grouped cross-validation on train (mean +/- std), AUC per manipulation (each fake kind against all
  test reals), the confusion matrix of the best model.
- **E3:** the same models scored on the FF++ c40 test split (heavier compression).
- **E4:** the same models scored on the Celeb-DF test list (a dataset never trained on).
- **E6:** permutation importance of the best ABCD model per input and per feature group, and a
  group ablation.

How the numbers are protected from leakage is in [../METHOD.md](../METHOD.md) sections 12 and 16.
