"""One normalized objective for initialization, search and learning."""
from statistics import mean

def partition_score(report):
    """Scale-normalized score, always subordinate to independent feasibility."""
    units = list(report['units'].values())
    free = sum(u['area'] for u in units) + report['corridor_area']
    # Do not spend shared circulation area chasing a sub-percent area match.
    deadband=report.get('area_balance_deadband',0.)
    return (10 - 8 * mean([max(0.,u['area_error_ratio']-deadband) for u in units])
            - 6 * report['corridor_area'] / free
            - .08 * mean([max(0, u['corners'] - 4) for u in units])
            - .12 * mean([u.get('short_edges', 0) for u in units])
            + .4 * min(2., min(u['facade_length'] / u['required_facade_length'] for u in units))
            + .6 * min(u.get('daylight_coverage_proxy', 1.) for u in units)
            + .3 * report['structure_alignment_ratio']
            + .8 * report.get('consistent_reference_alignment_ratio',0.))

