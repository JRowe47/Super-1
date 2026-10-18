# Fixed-step adaptive depth pilot

Completed five-arm, three-seed Tiny Shakespeare byte-language experiment. Token-conditioned sampling produced a small improvement over uniform sampling, but failed the preregistered useful-effect and uncertainty gates. Dense RoPE had better quality and lower uncached latency. No architecture promotion.

- [Full report](../../reports/adaptive-depth-shakespeare-20261018-v1.md)
- [Frozen protocol](PROTOCOL.json)
- [Raw results](execution/RESULT.json)
- [Verification](execution/VERIFICATION.json)
- [Terminal receipt](execution/TERMINAL.json)

The model learns two internal sampling positions per token while preserving four executed blocks. It retains the original cubic projection weight function and per-block normalization. All routing decisions come from a trained controller; there are no token-category rules.

Corpus: Tiny Shakespeare from Andrej Karpathy's char-rnn repository, a collection of public Shakespeare text. The exact source URL and content hash are in PROTOCOL.json; corpus.txt is the frozen snapshot. None of the historical sealed benchmark files was used.

Original training implementation and protocol are frozen. verify_results.py independently recomputes all 15 confirmation endpoints without training. report.py generates figures from saved metrics; it refuses to replace an existing report. All training checkpoints are included.

For future replay tooling, note that the frozen runner's artifact-size check scans its own run directory. It covered the original execution correctly because execution/ is inside that directory. An external --output replay also needs external-directory storage accounting. No historical receipt is rewritten to change this implementation detail.
