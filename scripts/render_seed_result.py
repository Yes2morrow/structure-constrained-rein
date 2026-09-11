"""Export an existing seed result directly into its training run directory."""

# Allow direct script execution from any working directory.
import sys
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import yaml
from core.seed_growth.rendering import render_result

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args(); run=args.run_dir.resolve()
    pointer=json.loads((run/'latest_precise.json').read_text(encoding='utf-8'))
    source=(run/pointer['path']).resolve()
    if run not in source.parents: raise ValueError('Result escapes run directory')
    result=json.loads(source.read_text(encoding='utf-8'))
    config=yaml.safe_load((run/'config.yaml').read_text(encoding='utf-8'))
    output=args.output or run
    print(render_result(config,result,output))
