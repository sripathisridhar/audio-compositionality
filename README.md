# This is a fork to evaluate modern audio representations.


# audio-compositionality

[![arXiv](https://img.shields.io/badge/arXiv-2603.13685-b31b1b.svg)](https://arxiv.org/abs/2603.13685)
[![Dataset: A-TRE-10k](https://img.shields.io/badge/🤗-A--TRE--10k-yellow)](https://huggingface.co/datasets/chuyangchenn/a-tre-10k)
[![Dataset: A-COAT-2k](https://img.shields.io/badge/🤗-A--COAT--2k-yellow)](https://huggingface.co/datasets/chuyangchenn/a-coat-2k)

This repository accompanies the ICASSP 2026 paper:

[**Evaluating Compositional Structure in Audio Representations**](https://arxiv.org/abs/2603.13685)  
Chuyang Chen, Bea Steers, Brian McFee, Juan Pablo Bello

It provides two benchmarks for evaluating whether an audio encoder
represents multi-source scenes compositionally:

- **A-COAT** (Audio Compositional Object Algebra Test) — zero-shot. Score is
  `cos(z_B − z_A, z_D − z_C)` over quadruples where `B = A ∪ T` and `D = C ∪ T`.
- **A-TRE** (Audio Tree Reconstruction Error) — trains a small composition
  head on top of frozen encoder embeddings. Score is `cos(z, ẑ)`.

Both metrics live in `[−1, 1]`; higher is better.

## Install

Python 3.10+. From a clean checkout:

```bash
git clone https://github.com/chuyangchencd/audio-compositionality
cd audio-compositionality
pip install -e .
```

## Quick start

The shipped baselines (`random`, `downsample`) need nothing beyond the
install. Datasets are pulled from HuggingFace on first use; encoder
embeddings are computed and cached after the first run.

```bash
# A-COAT — zero-shot, prints the score and writes outputs/random/a_coat_results.json
python cli/eval_a_coat.py configs/random.yaml

# A-TRE — train the composition head, then evaluate on the test split
python cli/train_a_tre.py configs/random.yaml
python cli/eval_a_tre.py  configs/random.yaml
```

Override any config value inline:

```bash
python cli/eval_a_coat.py configs/random.yaml train.batch_size=128 cache.dir=/fast/disk/cache
```

## Datasets

Two HuggingFace datasets, fetched automatically by `load_dataset`:

| Repo | Splits | Per-row schema |
|---|---|---|
| [`chuyangchenn/a-tre-10k`](https://huggingface.co/datasets/chuyangchenn/a-tre-10k) | train (8000), val (1000), test (1000) | `audio` (32 kHz mono), `metadata` (list of source-attribute dicts) |
| [`chuyangchenn/a-coat-2k`](https://huggingface.co/datasets/chuyangchenn/a-coat-2k) | test (2000) | `A`, `B`, `C`, `D` (each 32 kHz mono), `metadata` (`{A, C, T}` source dicts) |

Each source has four discrete attributes (8 classes each):

- **timbre** `t1`–`t8` — DX7 FM synth patches
- **pitch** `p1`–`p8` — MIDI 36–84, linearly binned
- **rate** `r1`–`r8` — 0.2–3.0 Hz repetition rate, log-binned
- **amplitude** `a1`–`a8` — −26 to 0 dB, linearly binned

Streaming works without local download:

```python
from datasets import load_dataset
ds = load_dataset("chuyangchenn/a-tre-10k", split="test", streaming=True)
for ex in ds.take(5):
    print(ex["audio"].get_all_samples().data.shape)
```

## Architecture

```
audiocomp/
├── data/        # PyTorch Dataset adapters around load_dataset
├── models/      # AudioEncoder ABC + baselines (random, downsample)
├── tasks/       # A-COAT scorer + A-TRE training/eval (operate on cached embeddings)
├── cache.py     # EmbeddingCache + CachedDatasetView (disk-persisted)
├── config.py    # YAML loader with deep-merge + dot-path interpolation
└── cli_utils.py # Shared parse_args + auto-build cache helper
cli/             # Three thin CLI wrappers: eval_a_coat, train_a_tre, eval_a_tre
configs/         # base.yaml + per-encoder overrides
```

The encoder–dataset–task contract:

- **Encoders** subclass `AudioEncoder` (in `audiocomp/models/base.py`) and
  expose `sample_rate`, `embedding_dim`, and `embed((B, T) → (B, D))`.
  Construction args (checkpoints, hyperparameters) live in the encoder's
  own `__init__` — the framework just passes through whatever's in the
  YAML's `model:` section.
- **Datasets** auto-resample audio to the encoder's `sample_rate` (HF
  feature-level cast) and yield canonical shapes: `(T,)` for A-TRE,
  `(4, T)` for A-COAT.
- **Tasks** consume only `CachedDatasetView` (which yields embeddings
  rather than waveforms) — they never see the encoder, never decode audio.
- **Caching** is automatic: the first eval run builds
  `<cache.dir>/<model>/<dataset>-<split>.pt`, every subsequent run reads it.
  Fingerprinted by `(model_name, sample_rate, embedding_dim, init_kwargs)`;
  changing any of these triggers `CacheFingerprintMismatch` so you don't
  silently use stale embeddings.

## Adding your own encoder

Three small edits:

1. Create `audiocomp/models/my_model.py`:

   ```python
   from .base import AudioEncoder

   class MyEncoder(AudioEncoder):
       sample_rate = 16000
       embedding_dim = 512

       def __init__(self, checkpoint=None):
           self._model = ...                # build / load weights
           if checkpoint:
               self._model.load_state_dict(torch.load(checkpoint))
           self._model.eval()

       def embed(self, audio):              # audio: (B, T) at sample_rate
           return self._model(audio)        # → (B, embedding_dim)
   ```

2. Register it in `audiocomp/models/__init__.py` — add the import next
   to the existing ones, and add a key to `_REGISTRY`:

   ```python
   from .baselines import DownsampleEncoder, RandomEncoder
   from .my_model import MyEncoder                        # ← add

   _REGISTRY: dict[str, type[AudioEncoder]] = {
       "random": RandomEncoder,
       "downsample": DownsampleEncoder,
       "my_model": MyEncoder,                             # ← add
   }
   ```

3. Create `configs/my_model.yaml`:

   ```yaml
   model:
     name: my_model
     checkpoint: /path/to/weights.pt   # optional; forwarded to MyEncoder.__init__
   ```

Then `python cli/eval_a_coat.py configs/my_model.yaml` etc. work the same
way as for the shipped baselines. The cache logic, padding, resampling and
batch shapes are model-agnostic.

## Citation

```bibtex
@inproceedings{chen2026audiocomp,
  title         = {Evaluating Compositional Structure in Audio Representations},
  author        = {Chen, Chuyang and Steers, Bea and McFee, Brian and Bello, Juan Pablo},
  booktitle     = {IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year          = {2026},
  eprint        = {2603.13685},
  archivePrefix = {arXiv},
  primaryClass  = {cs.SD}
}
```

## License

Code: [MIT](LICENSE).
Datasets: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
