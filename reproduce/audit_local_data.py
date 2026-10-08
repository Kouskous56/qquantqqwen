"""Read-only inventory of a Qwen lab; write only to an explicit report path.

Large weights are listed by size unless --hash-models is supplied. No file
contents, credentials, or model weights are copied into the report.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory(root, *, hash_models=False):
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"Lab directory does not exist: {root}")
    records = []
    # Explicit project directories: never sweep a home folder or Ollama keys.
    directories = ("bench", "scripts", "notes", "report", "papers", "models")
    paths = [p for p in root.iterdir() if p.is_file() and not p.is_symlink()]
    for name in directories:
        directory = root / name
        if directory.is_symlink():
            continue
        if directory.is_dir():
            paths.extend(p for p in directory.rglob("*") if p.is_file())
    for path in sorted(paths):
        relative = path.relative_to(root)
        if any(part in {".git", "__pycache__", ".pytest_cache"} for part in relative.parts):
            continue
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            continue
        record = {"path": relative.as_posix(), "bytes": path.stat().st_size}
        large_model = path.suffix.lower() in {".gguf", ".bin", ".safetensors", ".pt", ".pth"}
        if not large_model or hash_models:
            record["sha256"] = file_sha256(path)
        if path.suffix.lower() == ".json":
            try:
                json.loads(path.read_text(encoding="utf-8-sig"))
                record["json_valid"] = True
            except (UnicodeError, json.JSONDecodeError) as exc:
                record["json_valid"] = False
                record["error"] = str(exc)
        records.append(record)
    return records


def compare_release(repo, release):
    """Compare shared project files without touching the release checkout."""
    rows = []
    for source in sorted(Path(repo).rglob("*")):
        if not source.is_file() or source.is_symlink():
            continue
        rel = source.relative_to(repo)
        if any(p.startswith(".") or p == "__pycache__" for p in rel.parts):
            continue
        local = Path(release) / rel
        if local.is_file() and not local.is_symlink():
            row = {"path": rel.as_posix(),
                   "same_bytes": file_sha256(source) == file_sha256(local)}
            if source.suffix.lower() in {".py", ".md", ".json", ".csv", ".txt", ".cff", ".yml"}:
                try:
                    left = source.read_text(encoding="utf-8-sig")
                    right = local.read_text(encoding="utf-8-sig")
                    row["same_normalized_text"] = left == right
                    if source.suffix.lower() == ".json":
                        row["same_json"] = json.loads(left) == json.loads(right)
                except (UnicodeError, json.JSONDecodeError):
                    pass
            rows.append(row)
    return rows


def verify_models(records, manifest_dir):
    models = {Path(r["path"]).name: r for r in records if r["path"].startswith("models/")}
    checks = []
    for path in sorted(Path(manifest_dir).glob("RUN_*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        provenance = manifest.get("provenance", {})
        name = provenance.get("gguf")
        if not name:
            continue
        # Historical paths may use Windows separators on any host.
        name = name.replace("\\", "/").rsplit("/", 1)[-1]
        model = models.get(name)
        expected = provenance.get("gguf_sha256")
        actual = model.get("sha256") if model else None
        checks.append({"manifest": path.name, "model": name,
                       "available_now": model is not None,
                       "expected_sha256": expected, "actual_sha256": actual,
                       "hash_matches": (actual == expected) if actual and expected else None})
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lab-root", required=True, type=Path)
    parser.add_argument("--release-root", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--hash-models", action="store_true")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    for source in (args.lab_root, args.release_root):
        if source is not None and args.out.resolve().is_relative_to(source.resolve()):
            parser.error("Write the audit outside the source lab/release directories")
    records = inventory(args.lab_root, hash_models=args.hash_models)
    report = {
        "lab_root": str(args.lab_root.resolve()),
        "scope": "bench, scripts, notes, report, papers, models, and root files; tools excluded",
        "model_hashes_computed": args.hash_models,
        "counts_by_directory": dict(Counter(r["path"].split("/")[0] for r in records)),
        "total_files": len(records), "total_bytes": sum(r["bytes"] for r in records),
        "files": records,
        "model_verification": verify_models(records, args.repo_root / "manifests_qwen3"),
    }
    if args.release_root:
        if not args.release_root.is_dir():
            parser.error("Release directory does not exist")
        report["release_comparison"] = compare_release(args.repo_root, args.release_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Inventoried {len(records)} files; report: {args.out}")


if __name__ == "__main__":
    main()
