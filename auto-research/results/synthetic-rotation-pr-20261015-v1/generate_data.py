"""Generate independent synthetic partitions using only Python's standard library."""
import json
import random
from pathlib import Path


def generate(config):
    data = {}
    for split, seed in config['split_generator_seeds'].items():
        rng = random.Random(seed)
        batches = []
        for _ in range(config['split_batches'][split]):
            batch = []
            for _ in range(config['batch_size']):
                motif = [rng.randrange(config['vocab_size']) for _ in range(config['motif_length'])]
                batch.append([motif[i % len(motif)] for i in range(config['sequence_length'] + 1)])
            batches.append(batch)
        data[split] = batches
    return data


if __name__ == '__main__':
    here = Path(__file__).resolve().parent
    config = json.loads((here / 'PROTOCOL.json').read_text())['config']
    print(json.dumps(generate(config), separators=(',', ':')))
