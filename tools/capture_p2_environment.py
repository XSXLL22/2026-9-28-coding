"""Record installed package versions and local compute capabilities."""
import json
import platform
import subprocess
from importlib.metadata import distributions
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runtime.common import environment, write_json


def main():
    output = ROOT / "experiments" / "p2_environment"
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "environment.json", environment())
    installed = sorted({f"{d.metadata['Name']}=={d.version}" for d in distributions()}, key=str.lower)
    (ROOT / "requirements-p2.lock.txt").write_text("\n".join(installed) + "\n", encoding="utf-8")
    print(json.dumps(environment(), ensure_ascii=False))


if __name__ == "__main__":
    main()
