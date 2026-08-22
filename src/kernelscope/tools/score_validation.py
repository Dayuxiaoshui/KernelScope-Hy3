from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..metrics import LocalizationSample, evaluate_localization

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.predictions.read_text())
    rows = [LocalizationSample(**row) for row in data["samples"]]
    print(json.dumps(evaluate_localization(rows), indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
