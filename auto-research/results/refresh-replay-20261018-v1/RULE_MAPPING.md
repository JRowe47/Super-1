# Causal bank refresh

The learner and complete Fourier representation remain byte-identical to the predecessor. See BASELINE_RULE_MAPPING.md, feedback_core.py and replay_core.py. Warmup, spectral cap2, stateless SGD, lr0.05, feedback rate0.8 and weight decay are unchanged. No exact hidden gradient enters local updates.

The first256 of the common initial512 warm examples form a bank; the remaining256 form a rolling queue. At online time t, draw the same16 replay integers in[0,512) as the permanent mixed control. The first8 modulo256 index the recent queue; the last8 modulo256 index the bank. Train on16 current+8 recent+8 bank examples.

After sampling, append the current16 to the recent queue and discard its oldest16. Independently draw j uniformly from0..15 using seed+66,000,000; overwrite bank slot t modulo256 with the current pair (x[t,j], y[t,j]). Replacement is equivalent to occurring after the model update because it cannot affect that update and depends on no model state. Preserve the label as observed. Repeat every update regardless of phase, context, loss or correctness.

All512 rows count toward capacity, including a recent/bank duplicate. FIFO and permanent mixed controls are byte-preserved. After256 online updates all bank rows have online ages1..256 before sampling. Initial warm examples have a bookkeeping birth step-1, not a claim about their real age. Stale-target counts use known generator labels only as an audit and never control sampling, replacement or learning.

The confirmatory comparison is adaptive_refreshed versus frozen_refreshed. Other contrasts are descriptive. Refresh changes both label staleness and bank content/age; it does not isolate relabeling or stale-label removal alone.
