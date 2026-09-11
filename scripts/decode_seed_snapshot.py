"""Resume offline precise output from an already saved seed snapshot."""

# Allow direct script execution from any working directory.
import sys
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import yaml
from core.seed_growth.contracts import build_problem
from core.seed_growth.artifacts import RunArtifacts,read_snapshot


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--snapshot',type=Path)
    args=parser.parse_args()
    config=yaml.safe_load((args.run_dir/'config.yaml').read_text(encoding='utf-8'))
    problem=build_problem(config)
    data=json.loads((args.snapshot or args.run_dir/'latest_snapshot.json').read_text(encoding='utf-8'))
    snapshot=read_snapshot(problem,data.get('snapshot',data))
    result=RunArtifacts(args.run_dir,config,problem).emit(snapshot,'interrupted')
    print(json.dumps(dict(status=result['status'],snapshot_key=snapshot.key),ensure_ascii=False))
    if result['status']=='decode_failed': raise SystemExit(1)


if __name__=='__main__': main()
