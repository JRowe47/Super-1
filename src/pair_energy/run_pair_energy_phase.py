"""Fresh matched-budget comparison of relative pair-energy transport."""
import hashlib
from pathlib import Path

import run_overlapping_phase as runner
import pair_energy_phase_candidate as energy
import relative_phase_key_candidate as keyed

original_checks = runner.candidate.checks


def builders(config):
    width = config['width']
    target = runner.pilot.count(energy.PairEnergyPhaseLM(width))
    upper = 2
    while runner.pilot.count(runner.pilot.TinyTransformer(upper)) < target:
        upper *= 2
    d = min(range(2, upper + 1, 2),
            key=lambda v: abs(runner.pilot.count(runner.pilot.TinyTransformer(v)) - target))
    ff = min(range(d, 4*d + 1),
             key=lambda v: abs(runner.pilot.count(runner.pilot.TinyTransformer(d, v)) - target))
    makes = {
        'pair_energy': lambda: energy.PairEnergyPhaseLM(width),
        'geometric_overlapping': lambda: runner.candidate.OverlappingPhaseLM(width, True),
        'relative_phase_key': lambda: keyed.RelativePhaseKeyLM(width),
        'transformer': lambda: runner.pilot.TinyTransformer(d, ff),
    }
    return {name: makes[name] for name in config['arms']}, d, ff


def checks():
    return {'overlapping_checks': original_checks(), 'pair_energy_checks': energy.checks(),
            'phase_key_checks': keyed.checks(),
            'source_hashes': {str(Path(path).resolve()): hashlib.sha256(Path(path).read_bytes()).hexdigest()
                              for path in [__file__, energy.__file__, keyed.__file__]},
            'hypothesis': 'Relative pair energy augments oriented geometric queries and improves reserved CE over both geometry controls in all three fresh seeds.',
            'limitation': 'No information-capacity, contraction, original-hierarchy, compute-efficiency, novelty or SOTA claim.'}


if __name__ == '__main__':
    runner.builders = builders
    runner.candidate.checks = checks
    runner.main()
