# PulseCheck

Detecting deepfakes by checking for a real heartbeat (rPPG) in facial video.

Status: early development. This is a stub; full documentation will follow.

## Setup

```powershell
py -3.13 -m venv C:\dev\venvs\pulsecheck
C:\dev\venvs\pulsecheck\Scripts\python -m pip install -e ".[dev]"
$env:PULSECHECK_DATA = "C:\dev\pulsecheck-data"   # models and caches live outside the repo
C:\dev\venvs\pulsecheck\Scripts\python -m pytest -q
```

## License

MIT
