"""Publish checked recorder files without replacing existing dataset contents."""

import argparse
import hashlib
import json
import re
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
from huggingface_hub.hf_api import RepoFile


def validate_attributes(old, new, added_lfs_paths):
    permitted = {f"{name} filter=lfs diff=lfs merge=lfs -text" for name in added_lfs_paths}
    old_lines = iter(old.splitlines())
    previous = next(old_lines, None)
    for line in new.splitlines():
        if line == previous:
            previous = next(old_lines, None)
        elif line not in permitted:
            raise ValueError("An unrelated Git attribute changed")
    if previous is not None:
        raise ValueError("An existing Git attribute was removed")


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--checks", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--subsets", nargs="+", required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preupload-mcap-only", action="store_true")
    mode.add_argument("--audit-revision", help="Verify an existing immutable commit; never publish")
    args = parser.parse_args()
    if not args.preupload_mcap_only and not args.receipt:
        parser.error("--receipt is required when publishing")
    api = HfApi(endpoint="https://huggingface.co", token=True)
    info = api.dataset_info(args.repo)
    if info.private is not True or info.id != args.repo:
        raise ValueError("Expected the existing private dataset")

    def tree(revision):
        return {
            entry.path: entry
            for entry in api.list_repo_tree(
                args.repo, recursive=True, repo_type="dataset", revision=revision
            )
            if isinstance(entry, RepoFile)
        }

    def verify(operation, entry):
        local = operation.upload_info
        if entry.size != local.size:
            raise ValueError(f"Hub size differs: {entry.path}")
        if entry.lfs:
            matches = entry.lfs.sha256 == local.sha256.hex()
        else:
            prefix = f"blob {local.size}\0".encode()
            digest = hashlib.sha1(prefix)
            with (args.raw / entry.path).open("rb") as stream:
                for chunk in iter(lambda: stream.read(2**20), b""):
                    digest.update(chunk)
            matches = digest.hexdigest() == entry.blob_id
        if not matches:
            raise ValueError(f"Hub content differs: {entry.path}")

    manifest = hashlib.sha256(args.checks.read_bytes()).hexdigest()
    scope = dict(repo=args.repo, manifest_sha256=manifest)
    pin_file = args.receipt.with_suffix(".publish.json") if args.receipt else None
    pin = json.loads(pin_file.read_text()) if pin_file and pin_file.exists() else None
    if pin and any(pin.get(key) != value for key, value in scope.items()):
        raise ValueError("Commit receipt belongs to a different dataset or source")
    if args.audit_revision:
        history = api.list_repo_commits(
            args.repo, repo_type="dataset", revision=args.audit_revision
        )
        if len(history) < 2 or history[0].commit_id != args.audit_revision:
            raise ValueError("Expected an exact dataset commit with a parent")
        if pin and pin["revision"] != args.audit_revision:
            raise ValueError("Existing commit receipt points to another revision")
        pin = dict(scope, parent=history[1].commit_id, revision=args.audit_revision)
    parent = pin["parent"] if pin else info.sha
    before = {} if args.preupload_mcap_only else tree(parent)
    operations = {}
    for line in args.checks.read_text().splitlines():
        digest, name = line.split("  ", 1)
        path = Path(name)
        if args.preupload_mcap_only and (
            path.parts[0] not in args.subsets or path.suffix != ".mcap"
        ):
            continue
        if (
            path.is_absolute()
            or ".." in path.parts
            or len(path.parts) < 3
            or path.parts[0] not in args.subsets
            or not re.fullmatch(r"episode_\d{3}", path.parts[1])
            or name in operations
            or (args.raw / path).is_symlink()
        ):
            raise ValueError(f"Unexpected recorder path: {name}")
        op = CommitOperationAdd(path_in_repo=name, path_or_fileobj=args.raw / path)
        if op.upload_info.sha256.hex() != digest:
            raise ValueError(f"Local copy differs from recorder: {name}")
        if name in before:
            verify(op, before[name])
        operations[name] = op
    if args.preupload_mcap_only:
        if {Path(name).parts[0] for name in operations} != set(args.subsets):
            raise ValueError("No checked MCAP for a requested subset")
        api.preupload_lfs_files(
            args.repo, list(operations.values()), repo_type="dataset", num_threads=3
        )
        print(f"Staged {len(operations)} verified MCAPs; no Hub commit created", flush=True)
        return
    if set(operations) != {
        p.relative_to(args.raw).as_posix() for p in args.raw.rglob("*") if p.is_file()
    }:
        raise ValueError("Local file set differs from the recorder manifest")
    episodes = {str(Path(name).parents[1]) for name in operations if name.endswith(".mcap")}
    if not episodes or episodes != {"/".join(name.split("/")[:2]) for name in operations}:
        raise ValueError("Missing MCAP in recorder manifest")
    print(f"Verified local copy: {len(episodes)} episodes, {len(operations)} files", flush=True)
    pending = [op for name, op in operations.items() if name not in before]
    revision = pin["revision"] if pin else info.sha
    if pending and not pin:
        revision = api.create_commit(
            repo_id=args.repo,
            repo_type="dataset",
            parent_commit=info.sha,
            operations=pending,
            num_threads=4,
            commit_message=f"Add {len(episodes)} finalized {', '.join(args.subsets)} trajectories",
        ).oid
        pin = dict(scope, parent=parent, revision=revision)
    if pin:
        write_json(pin_file, pin)
    after = tree(revision)
    for name, op in operations.items():
        verify(op, after[name])
    changes = {
        name
        for name, old in before.items()
        if name not in after or old.blob_id != after[name].blob_id
    }
    if (
        changes - {".gitattributes"}
        or ".gitattributes" in changes
        and ".gitattributes" not in after
    ):
        raise ValueError("An existing dataset file changed")
    if changes:

        def attributes(ref):
            filename = hf_hub_download(
                args.repo,
                ".gitattributes",
                repo_type="dataset",
                revision=ref,
                token=True,
                endpoint=api.endpoint,
            )
            return Path(filename).read_text()

        validate_attributes(
            attributes(parent),
            attributes(revision),
            {name for name in operations if name not in before and after[name].lfs},
        )
    if pin and set(after) - set(before) != {op.path_in_repo for op in pending}:
        raise ValueError("Unexpected file added in this dataset commit")
    final = api.dataset_info(args.repo, revision=revision)
    if not final.private or final.sha != revision:
        raise ValueError("Final private dataset revision did not match")
    receipt = dict(
        repo=info.id,
        private=True,
        revision=revision,
        parent_at_start=parent,
        episodes=sorted(episodes),
        files=len(operations),
        previous_files=len(before),
        bytes=sum(op.upload_info.size for op in operations.values()),
        manifest_sha256=manifest,
    )
    write_json(args.receipt, receipt)
    print(f"Verified Hub revision {revision}", flush=True)


if __name__ == "__main__":
    main()
