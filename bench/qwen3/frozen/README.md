# Published source snapshots

These six files preserve the exact pre-refinement source bytes, named by their
SHA-256 digest. Published manifests remain unchanged. Their `current_sha256`
describes the source at publication and resolves to a file here.

The snapshots are historical evidence, not supported runners. Their known
checkpoint and portability defects remain intact so their hashes remain true.
Use the named scripts one directory above for new evaluations. New runs record
their arguments, dataset content digest and both runner/common source digests.
Legacy checkpoints do not contain that identity and are rejected: retain them
for evidence and choose a new `--tag` when evaluating with the maintained code.

An archive matching `current_sha256` does **not** establish that it generated
the historical scores. Where `scorer_sha256` differs, that older runtime source
may still be unavailable. Tests validate only the exact bytes actually present;
they neither regenerate model outputs nor reattribute frozen scores.

GSM extraction retains the historical string comparison (including decimal
formatting limitations). MMLU now accepts lowercase answer letters without
raising `ValueError`. These changes do not rescore published manifests.
