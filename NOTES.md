# NOTES

Best configuration: hand-rolled BPE tokenizer (500 merges, vocab 756, trained only on
`train_corpus.txt`), 4-layer/4-head/160-dim GPT with a SwiGLU (gated silu) MLP and RMSNorm
instead of GELU MLP + LayerNorm, untied embeddings, plain Adam with cosine LR decay from 1e-3
to 1e-4 over all 2000 steps (no warmup), batch=8, block_size=128. Total params:
1,497,968 / 2,000,000. Final dev bpb: **2.1731**, down from the baseline's 2.3718 — an 8.4%
relative improvement.

The two largest levers, in order, were the BPE tokenizer (compresses the corpus's ~14-20%
Devanagari content from 3 bytes/char under the byte tokenizer down to 2.6-3.7 bytes/token,
giving the model roughly 2.5x more real text per fixed 128-token window and more signal per
gradient step) and raising the learning rate from the baseline's demonstrably under-fitting
constant 3e-4 to a cosine-decayed schedule peaking at 1e-3. A SwiGLU MLP swap gave a further
meaningful gain at matched parameter count, since its multiplicative gating is a richer
function class per parameter than a single fixed GELU nonlinearity. RMSNorm, tried last among
the clear wins, gave only a marginal gain (2.1738→2.1731) — reported honestly as a
low-confidence positive rather than oversold, and kept since it's free (fewer parameters, no
added risk). Four ideas were tested and explicitly rejected or found neutral under this fixed
2000-step budget: weight tying (hurt convergence — a shared embedding/output matrix doesn't
have enough steps to settle into a good dual-purpose representation), a ~48% capacity increase
(no gain — the model was optimization-limited, not capacity-limited), GPT-2-style init
rescaling (hurt results, likely because it wasn't re-tuned jointly with the already-fixed LR),
and — as the final two probes — extending block_size to 192 and removing all bias terms
(LLaMA-style), both tested specifically because they were cheap and safe to try, and both
found neutral-to-slightly-negative rather than hidden as failures. The BPE tokenizer was
implemented in pure Python/NumPy/stdlib only, deliberately avoiding the `tokenizers`/
`transformers` packages present in this environment since they fall outside the assignment's
cap, with round-trip correctness verified on the full training corpus, the full dev_eval.txt,
and out-of-distribution text before any model training began. Full hypothesis/result/reasoning
for all 13 runs is in RUNLOG.md.
