from __future__ import annotations

from pathlib import Path
import sys


EXPERIMENT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(PROJECT_ROOT), str(EXPERIMENT_ROOT)]

from index_aware_codec.run_replay import main


if __name__ == "__main__":
    main()
