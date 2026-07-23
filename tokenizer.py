"""Run 10: hand-rolled byte-pair-encoding (BPE) tokenizer, trained only on
train_corpus.txt (pure Python/NumPy/stdlib - no HuggingFace `tokenizers`,
which would violate the "pure PyTorch + numpy + stdlib" cap even though it
happens to be installed in this environment).

Base vocab is the full byte range (0-255), so encode() always succeeds on
arbitrary UTF-8 text by construction - the byte fallback the assignment
requires is automatic, not a special case. Losslessness: merges only ever
combine two existing ids into one new id; decode() recursively expands each
id back to its constituent bytes, so decode(encode(text)) == text exactly
for any text encode() can produce ids for (i.e. always, since encode()
starts from raw UTF-8 bytes).

Train once:  python tokenizer.py --train ../data/train_corpus.txt
Then load() (no args) reads the saved merges relative to this file.
"""
import argparse
import json
import os

import numpy as np

_MERGES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bpe_merges.json")


class BPETokenizer:
    def __init__(self, merges=None):
        # merges: list of [[a, b], new_id] in the order they were trained
        self.merges = merges or []
        self.vocab_size = 256 + len(self.merges)
        self._expand = {new_id: (a, b) for (a, b), new_id in self.merges}

    def encode(self, text):
        ids = np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(np.int32)
        for (a, b), new_id in self.merges:
            ids = _merge_np(ids, a, b, new_id)
        return ids.tolist()

    def decode(self, ids):
        cache = {}

        def expand(tok):
            if tok < 256:
                return bytes([tok])
            if tok in cache:
                return cache[tok]
            a, b = self._expand[tok]
            result = expand(a) + expand(b)
            cache[tok] = result
            return result

        return b"".join(expand(t) for t in ids).decode("utf-8", errors="replace")


def _merge_np(arr, a, b, new_id):
    """Vectorized single-merge pass: replace every non-overlapping (a, b)
    with new_id. Overlaps are only possible when a == b (e.g. merging "aa"),
    since two adjacent matches at i and i+1 would otherwise require
    arr[i+1] to equal both a and b simultaneously."""
    if len(arr) < 2:
        return arr
    matches = (arr[:-1] == a) & (arr[1:] == b)
    idx = np.nonzero(matches)[0]
    if len(idx) == 0:
        return arr
    if a == b and len(idx) > 1:
        keep = np.ones(len(idx), dtype=bool)
        prev = idx[0]
        for i in range(1, len(idx)):
            if idx[i] == prev + 1:
                keep[i] = False
            else:
                prev = idx[i]
        idx = idx[keep]
    remove_mask = np.zeros(len(arr), dtype=bool)
    remove_mask[idx + 1] = True
    out = arr.copy()
    out[idx] = new_id
    return out[~remove_mask]


def _train(corpus_bytes, num_merges):
    arr = np.frombuffer(corpus_bytes, dtype=np.uint8).astype(np.int32)
    merges = []
    next_id = 256
    for m in range(num_merges):
        if len(arr) < 2:
            break
        left = arr[:-1].astype(np.int64)
        right = arr[1:].astype(np.int64)
        keys = left * 100_000 + right  # vocab stays far below 100,000
        uniq, counts = np.unique(keys, return_counts=True)
        best = uniq[np.argmax(counts)]
        a, b = int(best // 100_000), int(best % 100_000)
        arr = _merge_np(arr, a, b, next_id)
        merges.append(([a, b], next_id))
        if (m + 1) % 50 == 0:
            print(f"  merge {m+1}/{num_merges}: ({a},{b})->{next_id} "
                  f"count={counts.max():,} seq_len={len(arr):,}")
        next_id += 1
    return merges


def load(path=None):
    """Return the tokenizer used by train.py/evaluate.py. No required args."""
    path = path or _MERGES_PATH
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        merges = [([p[0], p[1]], nid) for p, nid in data["merges"]]
        return BPETokenizer(merges=merges)
    return BPETokenizer(merges=[])  # pure byte fallback if untrained


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="path to train_corpus.txt")
    ap.add_argument("--num_merges", type=int, default=500)
    ap.add_argument("--sample_bytes", type=int, default=2_000_000)
    args = ap.parse_args()

    full_text = open(args.train, encoding="utf-8").read()
    corpus_bytes = full_text.encode("utf-8")[:args.sample_bytes]
    print(f"training BPE on {len(corpus_bytes):,} bytes (of "
          f"{len(full_text.encode('utf-8')):,} total corpus), "
          f"{args.num_merges} merges")
    merges = _train(corpus_bytes, args.num_merges)

    with open(_MERGES_PATH, "w") as f:
        json.dump({"merges": merges}, f)
    print(f"saved {len(merges)} merges -> {_MERGES_PATH}")

    tok = BPETokenizer(merges=merges)
    test_text = full_text[:200_000]
    ids = tok.encode(test_text)
    back = tok.decode(ids)
    ok = back == test_text
    print(f"round-trip ok: {ok}  vocab_size: {tok.vocab_size}  "
          f"compression: {len(test_text.encode('utf-8'))/len(ids):.3f} bytes/token")
    if not ok:
        raise SystemExit("ROUND-TRIP FAILED on train sample - do not proceed")
