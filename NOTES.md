# NOTES

Best configuration: hand-rolled BPE tokenizer (500 merges, vocab 756, trained only on
`train_corpus.txt`), 4-layer/4-head/160-dim GPT with a SwiGLU (gated silu) MLP instead of
GELU, untied embeddings, plain Adam with cosine LR decay from 1e-3 to 1e-4 over all 2000 steps
(no warmup), no weight decay, no gradient clipping, batch=8, block_size=128. Total params:
1,499,408 / 2,000,000. Final dev bpb: **2.1738**, down from the baseline's 2.3718 — an 8.35%
relative improvement.

The two largest levers, in order, were: (1) the BPE tokenizer, which compresses the corpus's
~14-20% Devanagari content from 3 bytes/char under the byte tokenizer down to 2.6-3.7
bytes/token, giving the model roughly 2.5x more real text per fixed 128-token context window
and more signal per gradient step; and (2) raising the learning rate from the baseline's
constant 3e-4 (demonstrably under-fitting - loss still descending hard at step 2000) to a
cosine-decayed schedule peaking at 1e-3. A SwiGLU MLP swap, at essentially matched parameter
count, gave a further meaningful gain over the standard GELU MLP. Three other well-motivated
ideas were tested and explicitly rejected under this fixed step budget: weight tying (hurt
convergence - a shared embedding/output matrix doesn't have 2000 steps to settle into a good
dual-purpose representation), a ~48% capacity increase via a wider model (no gain - confirms
the model was optimization-limited, not capacity-limited), and GPT-2-style init rescaling
(hurt results, most likely because it wasn't re-tuned jointly with the already-fixed LR
schedule). The BPE tokenizer was implemented in pure Python/NumPy/stdlib only - the
`tokenizers`/`transformers` packages present in this environment were deliberately avoided
since they fall outside the "pure PyTorch + numpy + stdlib" cap - and its round-trip
correctness was verified on the full training corpus, the full dev_eval.txt, and
out-of-distribution text before any model training began, given that a lossy tokenizer
disqualifies the run. Full hypothesis/result/conclusion reasoning for all 10 runs is in
RUNLOG.md.
