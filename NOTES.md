# NOTES

Best configuration: unchanged byte-level tokenizer, 4-layer/4-head/160-dim GPT (1,339,840
params, untied embeddings), plain Adam with a cosine LR decay from 1e-3 to 1e-4 over all 2000
steps (no warmup), no weight decay, no gradient clipping, batch=8, block_size=128. Final dev
bpb: **2.2799**, down from the baseline's 2.3718 (3.9% relative improvement).

The dominant lever was learning rate: the baseline's constant 3e-4 was demonstrably
under-fitting (loss curve still descending hard at step 2000 with no plateau), and raising it
to 1e-3 alone produced by far the largest single gain of any change tested (-0.0876 bpb).
Weight tying was tested and rejected — isolated, it made results *worse*, likely because
forcing one matrix to serve both the input-embedding and output-classification roles doesn't
have enough of a 2000-step budget to settle into a good shared representation. Pushing model
capacity toward the 2M-parameter cap (widening to ~1.98M params) also did not help, confirming
the model was optimization-limited rather than capacity-limited at this step count. A cosine
LR decay on top of the proven 1e-3 peak (no warmup, since the constant-LR run was stable from
step 1) gave a further small improvement by consolidating late-training gains instead of
oscillating near a minimum. An AdamW + weight-decay + gradient-clipping "hygiene" bundle was
tried early and rejected — unnecessary regularization when there is no real overfitting risk at
under 30% data coverage in 2000 steps. The one high-ceiling idea not attempted was a BPE
tokenizer (the corpus's ~14-20% Devanagari content triples in length under byte-level
tokenization) — deliberately deprioritized given its round-trip-correctness disqualification
risk under a hard deadline; it remains the strongest follow-up with more time. Full
hypothesis/result/conclusion reasoning for all 7 runs is in RUNLOG.md.
