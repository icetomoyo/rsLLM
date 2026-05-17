# DeepSeek V4 Flash Official Test Vectors

These vectors are **vendored** from the [ds4 reference repo](https://github.com/antirez/ds4)
(`tests/test-vectors/` at upstream commit `ef0a490`, 2026-05-17). They were
originally captured from the official DeepSeek V4 Flash hosted API using
`deepseek-v4-flash`, greedy decoding, thinking disabled, and
`top_logprobs=20`. The hosted API does not expose full logits, so these
files store the best logprob slice the API provides.

## Why vendor instead of fetch?

The hosted DeepSeek API is the only source of ground-truth model output for
DS V4 Flash. We vendor the vectors so:

- the F006 / F005 numerical-parity gate is reproducible without needing a
  DeepSeek API key
- breakage from upstream API drift is visible as a deliberate update
- CI doesn't depend on outbound HTTPS

When ds4 updates the vectors upstream (which has happened during their
own development), re-run the vendor step:

```sh
cp /path/to/ds4/tests/test-vectors/{README.md,manifest.json,official.vec} \
   tests/dsv4-vectors/
cp /path/to/ds4/tests/test-vectors/official/*.json tests/dsv4-vectors/official/
cp /path/to/ds4/tests/test-vectors/prompts/*.txt tests/dsv4-vectors/prompts/
```

## Files

- `prompts/*.txt` — exact user prompts (5 prompts: 3 short, 2 long).
- `official/*.official.json` — official API continuations and top-20 logprobs.
- `official.vec` — compact C-test fixture (kept for byte-identity audit).
- `manifest.json` — index over the 5 prompts (id, length, step count).

## Acceptance gate (rsLLM v0.1.0 §F005)

For each of the 5 prompts, drive rsLLM's greedy decoder for the manifest's
`steps` count (4 for most, 1 for `short_reasoning_plain` → 17 steps total).
At each step:

- **Top-1 match**: the rsLLM-produced argmax token must equal the
  official top-1 token (hit rate = 100%)
- **Top-20 KL**: the top-20 logprob distribution KL divergence from the
  official distribution must be ≤ 1e-3

The 100% top-1 gate is the **core correctness gate** for v0.1.0.

## Provenance

License: MIT (inherited from the ds4 repo). See `NOTICE.md` at the rsLLM
repo root for the full attribution.
