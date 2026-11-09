import os
from pathlib import Path
HERE=Path(__file__).resolve().parent
DATA=Path(os.environ.get('TASK3_DATA_DIR',str(HERE.parent/'Assignment 1/assignment1_2026')))
ART=Path(os.environ.get('TASK3_ARTIFACTS_DIR',str(HERE/'artifacts'/'task3')))
ART.mkdir(parents=True,exist_ok=True)
