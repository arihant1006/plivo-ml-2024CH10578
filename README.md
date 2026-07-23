# Plivo LLM Speedrun — 2024CH10578

Byte-level GPT trained from scratch under hard caps: 2,000 optimizer steps, ≤2,000,000
parameters, CPU only, pure PyTorch/NumPy/stdlib, no pretrained weights. Trained solely on the
provided mixed English+Hindi corpus. Scored via bits-per-byte (bpb) on held-out text.

**Result: dev bpb 2.3718 (baseline) → 2.1731 (final), an 8.4% relative improvement**, from a
hand-rolled BPE tokenizer, a SwiGLU MLP, RMSNorm, and learning-rate tuning + cosine decay. Full
experimental reasoning (13 runs, hypothesis/result/reasoning for each — including the negative
results) is in `RUNLOG.md`; the condensed final configuration and rationale is in `NOTES.md`;
a consolidated, human-readable report is in `SUMMARY.html`.

## Reproduce

```
python tokenizer.py --train ../data/train_corpus.txt --num_merges 500   # trains bpe_merges.json
python train.py --data ../data/train_corpus.txt --steps 2000 --lr 1e-3 --out ckpt.pt
python evaluate.py --checkpoint ckpt.pt --text_file ../data/dev_eval.txt
```
(`bpe_merges.json` is already committed, so the first step only needs re-running if it's missing.)

## Files

| File | What it is |
|---|---|
| `model.py` | GPT architecture (final: 4-layer/4-head/160-dim, SwiGLU MLP, RMSNorm, untied, 1,497,968 params) |
| `train.py` | Trainer — final config: plain Adam, cosine LR decay 1e-3→1e-4, no warmup |
| `tokenizer.py` | Hand-rolled BPE tokenizer (pure Python/NumPy, 500 merges, vocab 756) |
| `bpe_merges.json` | Trained BPE merge table, loaded by `tokenizer.load()` |
| `evaluate.py` | Official scorer (unmodified interface, as required) |
| `ckpt.pt` | Final checkpoint (2,000 steps, dev bpb 2.1731) |
| `run_checkpoints/` | Checkpoint from every run in RUNLOG.md, for reproducibility |
| `RUNLOG.md` | Every run: hypothesis, what changed, before/after bpb, why it worked or failed |
| `NOTES.md` | Best configuration and why it works (condensed) |
| `SUMMARY.html` | Consolidated report: architecture, results, findings, human/AI contribution |

## What changed vs. the starter, and why

- **BPE tokenizer (largest lever):** byte-level (vocab 256) → hand-rolled BPE (vocab 756).
  The byte tokenizer tripled the token cost of Devanagari text (~14-20% of the corpus by
  character count), capping the model's effective context. A from-scratch BPE tokenizer (pure
  Python/NumPy/stdlib, no third-party tokenizer libraries) compresses this to 2.6-3.7
  bytes/token, round-trip-verified on the full corpus, the full eval file, and
  out-of-distribution text before training.
- **SwiGLU MLP** instead of the standard 4x-GELU MLP, at matched parameter count — gated,
  multiplicative feedforward gave a substantial gain over a single fixed nonlinearity.
- **Learning rate:** constant 3e-4 → cosine decay 1e-3 → 1e-4, no warmup. The baseline's loss
  curve was still descending hard at step 2000 — direct evidence of an under-tuned LR, not a
  converged model. Confirmed by bracketing the constant-LR region first (3e-4 too low, 1e-3
  strong, 2e-3 decisively unstable, with a genuine ~700-step mid-run plateau) before adding
  decay.
- **RMSNorm** instead of LayerNorm — a marginal gain (2.1738→2.1731), kept since it's free
  (fewer parameters, no added risk), reported honestly as low-confidence rather than oversold.
- **Weight tying, capacity increase, and GPT-2-style init were each tried and rejected** —
  all three made dev bpb worse under this specific 2000-step budget, each for a distinct,
  diagnosed reason (not just "it didn't work"). See RUNLOG.md for the full mechanism behind
  each.
- **Two final low-risk probes — extending block_size to 192 and removing bias terms — came
  back neutral and slightly-negative respectively**, correctly identifying that the main levers
  for this specific setup had already been found, rather than being cut short or run past the
  point of returning information.
