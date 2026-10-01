@echo off
rem PEB Gable Optimiser from source: needs Python 3.12+ on PATH; installs numpy / scipy / matplotlib once.
cd /d "%~dp0"
python -c "import numpy, scipy, matplotlib" 2>nul || python -m pip install --user -r requirements.txt
python app.py %*
