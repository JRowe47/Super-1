"""Fresh modular-accumulator documents, independent PRNG streams, no corpus."""
import random

def generate(config):
    data = {}
    for split, seed in config['split_generator_seeds'].items():
        rng = random.Random(seed)
        batches = []
        for _ in range(config['split_batches'][split]):
            batch = []
            for _ in range(config['batch_size']):
                state = rng.randrange(config['modulus'])
                doc = [state]
                for _ in range(config['readout_groups']):
                    for _ in range(config['commands_per_group']):
                        delta = rng.randrange(config['modulus'])
                        doc.append(config['modulus'] + delta)
                        state = (state + delta) % config['modulus']
                    doc.append(state)
                assert len(doc) == config['sequence_length'] + 1
                batch.append(doc)
            batches.append(batch)
        data[split] = batches
    docs = {split: {tuple(doc) for batch in batches for doc in batch}
            for split, batches in data.items()}
    for split, values in docs.items():
        assert len(values) == config['split_batches'][split] * config['batch_size']
        assert all(not values.intersection(other) for key, other in docs.items() if key != split)
    return data
