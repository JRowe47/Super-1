# Dataset attribution

The small token arrays in `data/token_windows.npz` are transformed excerpts
from WikiText-2 (raw), distributed by Salesforce as `Salesforce/wikitext`,
configuration `wikitext-2-raw-v1`. They contain Wikipedia article text authored
by Wikipedia contributors. Article titles and offsets for each sampled window
are recorded in `data/dataset_manifest.json`; these identify the underlying
articles and their attribution histories on English Wikipedia.

Dataset: https://huggingface.co/datasets/Salesforce/wikitext

Dataset paper: Stephen Merity, Caiming Xiong, James Bradbury, Richard Socher,
“Pointer Sentinel Mixture Models,” 2016. https://arxiv.org/abs/1609.07843

As described in the dataset card, WikiText is available under Creative Commons
Attribution-ShareAlike 3.0 and the GNU Free Documentation License.

- https://creativecommons.org/licenses/by-sa/3.0/
- https://www.gnu.org/licenses/fdl-1.3.html

This experiment joins article lines, tokenizes with the pinned SmolLM2
tokenizer, and samples nonoverlapping 129-token windows within each article.
The included derived token data retains the source attribution and licensing
requirements. This notice applies to source and derived dataset content,
not to experiment code or independent numerical results.

The pinned converted-Parquet revision, source checksums, per-split sample seeds,
selected article titles, offsets and token-array checksums are in the manifest.
The token arrays are sufficient to repeat inference; complete dataset downloads
are unnecessary unless regenerating the sample.
