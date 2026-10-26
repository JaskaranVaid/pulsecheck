# Heart-rate evaluation

**Pending: no real-data results exist yet.** UBFC-rPPG has not been downloaded and no own-clips
have been recorded, so there are no numbers here. Running the command below replaces this file with
the generated results table:

```powershell
$env:PULSECHECK_DATA = "C:\dev\pulsecheck-data"
C:\dev\venvs\pulsecheck\Scripts\python scripts/eval_hr.py `
    --ubfc-root C:\dev\pulsecheck-data\ubfc\DATASET_2 `
    --own-clips C:\dev\pulsecheck-data\own\clips.csv --max-seconds 60
```

See [../METHOD.md](../METHOD.md) for what is measured and how.
