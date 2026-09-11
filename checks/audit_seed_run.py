"""Read-only recovery audit: verify committed training and geometry evidence."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yaml
from shapely.geometry import shape
from core.seed_growth.contracts import build_problem
from core.seed_growth.validation import validate_layout


def audit(root):
    root = Path(root).resolve()

    def read(path):
        resolved = (root / path).resolve()
        if root not in resolved.parents:
            raise ValueError('Artifact path escapes run directory')
        return json.loads(resolved.read_text(encoding='utf-8'))

    status = read('run_status.json')
    checkpoint = read(read('latest_checkpoint.json')['path'] + '/checkpoint.json')
    result = read(read('latest_precise.json')['path'])
    snapshot = status['snapshot']
    assert checkpoint['snapshot'] == snapshot, 'Checkpoint differs from final snapshot'
    assert result['snapshot'] == snapshot, 'Precise output differs from final snapshot'
    metrics = [json.loads(line) for line in
               (root / 'estimated_metrics.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    episodes = [row['episode'] for row in metrics]
    assert episodes == list(range(1, snapshot['completed_episodes'] + 1)), 'Missing/duplicate episode metrics'
    config = yaml.safe_load((root / 'config.yaml').read_text(encoding='utf-8'))
    problem = build_problem(config)
    validation = validate_layout(problem, snapshot['seeds'],
                                 {key: shape(value) for key, value in result['polygons'].items()})
    assert validation == result['validation'], 'Independent validation differs from stored result'
    relations = validation['relations']
    adjacent = [edge for edge in relations if edge['kind'] == 'adjacent']
    return dict(run=str(root), status=status['status'], completed_episodes=snapshot['completed_episodes'],
                checkpoint_consistent=True, episode_log_contiguous=True,
                independently_revalidated=True,
                committed_precise_results=len(list((root / 'precise').glob('*/result.json'))),
                geometry_valid=validation['geometry_valid'],
                adjacent_satisfied=sum(edge['satisfied'] is True for edge in adjacent),
                adjacent_total=len(adjacent), unresolved_relations=[edge for edge in relations if edge['satisfied'] is not True],
                comparison=result['comparison'], precise_status=result['status'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dir')
    parser.add_argument('--output')
    args = parser.parse_args()
    report = json.dumps(audit(args.run_dir), ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(report + '\n', encoding='utf-8')
    print(report)
