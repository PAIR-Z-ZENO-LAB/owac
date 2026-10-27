"""Read-only LOWLIGHT report over Normal/Low/ExtremeLow teacher/student runs."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

DOMAINS = ("Normal", "Low", "ExtremeLow")
ROLES = ("teacher", "student")
EXPECTED = 20


def _wilson(k: int, n: int) -> list[float] | None:
    if n <= 0:
        return None
    z = 1.96
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0.0, mid - half), min(1.0, mid + half)]


def _episode_id(npz: Any, fallback: str) -> str:
    if "episode_id" in npz.files:
        return str(np.asarray(npz["episode_id"]).reshape(-1)[0])
    return fallback


def _read_role_dir(
    path: Path, expected: set[str]
) -> tuple[dict[str, dict[str, Any]], dict[str, str], list[dict[str, Any]]]:
    """Return completed ledger rows, error rows, and NPZ arrays by episode."""
    completed: dict[str, dict[str, Any]] = {}
    errors: dict[str, str] = {}
    audits: list[dict[str, Any]] = []
    ledger = path / "attempts.jsonl"
    if not ledger.is_file():
        return (
            completed,
            errors,
            [{"check": "ledger_present", "status": "fail", "detail": str(ledger)}],
        )
    for line in ledger.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        episode = str(row.get("episode_id", ""))
        if not episode:
            audits.append({"check": "ledger_episode_id", "status": "fail", "row": row})
            continue
        if episode not in expected:
            audits.append({"check": "unexpected_episode_id", "status": "fail", "episode": episode})
            continue
        if row.get("status") == "completed":
            if episode in completed:
                audits.append(
                    {"check": "duplicate_completed", "status": "fail", "episode": episode}
                )
            else:
                completed[episode] = row
        elif row.get("status") == "error":
            if episode in errors:
                audits.append({"check": "duplicate_error", "status": "fail", "episode": episode})
            else:
                errors[episode] = str(row.get("error", "error"))
    return completed, errors, audits


def _npz_map(path: Path, expected: set[str]) -> tuple[dict[str, Path], list[dict[str, Any]]]:
    result: dict[str, Path] = {}
    audits: list[dict[str, Any]] = []
    if not path.is_dir():
        return result, audits
    for file in sorted(path.glob("*.npz")):
        with np.load(file, allow_pickle=False) as archive:
            episode = _episode_id(archive, file.stem)
        if episode not in expected:
            audits.append({"check": "unexpected_npz", "status": "fail", "episode": episode})
        elif episode in result:
            audits.append({"check": "duplicate_npz", "status": "fail", "episode": episode})
        else:
            result[episode] = file
    return result, audits


def _decode_lighting(archive: Any) -> dict[str, Any] | None:
    if "lighting_json" not in archive.files:
        return None
    value = str(np.asarray(archive["lighting_json"]).reshape(-1)[0])
    return json.loads(value)


def _video_frame_count(path: Path) -> int | None:
    if not path.is_file() or shutil.which("ffprobe") is None:
        return None
    try:
        raw = (
            subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-count_frames",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=nb_read_frames",
                    "-of",
                    "csv=p=0",
                    str(path),
                ],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            .strip()
            .splitlines()[0]
        )
        return int(raw)
    except (OSError, subprocess.CalledProcessError, ValueError, IndexError):
        return None


def _audit_role(role: str, row: dict[str, Any] | None, path: Path | None) -> list[dict[str, Any]]:
    if row is None or path is None:
        return [{"check": f"present_{role}", "status": "missing"}]
    with np.load(path, allow_pickle=False) as archive:
        checks = []
        expected_actions = np.asarray(archive["physical"], dtype=np.float32)[:, :5].reshape(-1, 7)
        executed = np.asarray(archive["executed"], dtype=np.float32)
        expected_actions = expected_actions[: len(executed)]
        checks.append(
            {
                "check": f"executed_matches_physical_{role}",
                "status": "pass" if np.array_equal(expected_actions, executed) else "fail",
            }
        )
        if "success" in archive.files and "steps" in archive.files:
            checks.append(
                {
                    "check": f"success_steps_match_ledger_{role}",
                    "status": "pass"
                    if (
                        bool(np.asarray(archive["success"]).reshape(-1)[0])
                        == bool(row.get("success"))
                        and int(np.asarray(archive["steps"]).reshape(-1)[0])
                        == int(row.get("steps"))
                    )
                    else "fail",
                }
            )
        video = path.with_suffix(".mp4")
        frames = _video_frame_count(video)
        expected_frames = int(row.get("steps", -11)) + 11
        checks.append(
            {
                "check": f"video_frames_match_steps_{role}",
                "status": "unverifiable"
                if frames is None
                else "pass"
                if frames == expected_frames
                else "fail",
                "frames": frames,
                "expected": expected_frames,
            }
        )
        return checks


def _audit_pair(
    teacher: dict[str, Any] | None,
    student: dict[str, Any] | None,
    tpath: Path | None,
    spath: Path | None,
) -> dict[str, Any]:
    out: dict[str, Any] = {"episode": None, "checks": []}
    out["checks"].extend(_audit_role("teacher", teacher, tpath))
    out["checks"].extend(_audit_role("student", student, spath))
    if teacher is None or student is None or tpath is None or spath is None:
        out["checks"].append({"check": "pair_present", "status": "missing"})
        return out
    with np.load(tpath, allow_pickle=False) as ta, np.load(spath, allow_pickle=False) as sa:
        out["episode"] = _episode_id(ta, tpath.stem)
        for key in ("initial_state", "first_raw", "state_raw", "noise", "lighting_json"):
            if key not in ta.files or key not in sa.files:
                out["checks"].append({"check": key, "status": "unverifiable"})
                continue
            a = np.asarray(ta[key])
            b = np.asarray(sa[key])
            if key == "state_raw":
                a, b = a[:1], b[:1]
            if key == "noise":
                count = min(a.shape[0], b.shape[0])
                a, b = a[:count], b[:count]
            out["checks"].append(
                {
                    "check": key,
                    "status": "pass" if np.array_equal(a, b) else "fail",
                }
            )
    return out


def build_report(root: str | Path, *, expected: int = EXPECTED) -> dict[str, Any]:
    root = Path(root)
    catalog_path = root / "catalog.json"
    if not catalog_path.is_file():
        raise FileNotFoundError(f"lowlight catalog missing: {catalog_path}")
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    test_entries = [entry for entry in catalog.get("episodes", []) if entry.get("split") == "test"]
    expected_ids = [str(entry["id"]) for entry in test_entries]
    if len(set(expected_ids)) != len(expected_ids):
        raise ValueError("catalog test episode ids are not unique")
    expected_set = set(expected_ids)
    if len(expected_ids) != expected:
        raise ValueError(f"catalog has {len(expected_ids)} test episodes, expected {expected}")
    data: dict[tuple[str, str], dict[str, Any]] = {}
    for domain in DOMAINS:
        for role in ROLES:
            path = root / domain / role
            completed, errors, extra = _read_role_dir(path, expected_set)
            npzs, npz_audits = _npz_map(path, expected_set)
            extra = extra + npz_audits
            for episode in npzs:
                if episode not in completed:
                    errors.setdefault(episode, "npz_without_completed_ledger")
            data[(domain, role)] = {
                "path": path,
                "completed": completed,
                "errors": errors,
                "npzs": npzs,
                "extra_audits": extra,
            }
    report: dict[str, Any] = {
        "schema": "owac.raw-lowlight.report.v1",
        "expected_per_role": expected,
        "expected_episode_ids": expected_ids,
        "catalog": {"path": str(catalog_path), "test_episodes": len(expected_ids)},
        "conditions": {},
        "audits": [],
        "cross_domain_audits": [],
        "video_pairs": [],
    }
    success_by: dict[tuple[str, str], dict[str, bool]] = {}
    for domain in DOMAINS:
        condition: dict[str, Any] = {"roles": {}, "pairs": {}}
        for role in ROLES:
            record = data[(domain, role)]
            done = set(record["completed"]) & set(record["npzs"])
            missing = [episode for episode in expected_ids if episode not in done]
            error_ids = sorted(record["errors"])
            success = sum(bool(record["completed"].get(e, {}).get("success")) for e in done)
            condition["roles"][role] = {
                "success": success,
                "denominator": expected,
                "completed": len(done),
                "errors": len(error_ids),
                "missing": len(missing),
                "wilson": _wilson(success, expected),
                "error_ids": error_ids,
                "missing_ids": missing,
                "extra_audits": record["extra_audits"],
            }
            for extra in record["extra_audits"]:
                report["audits"].append({"domain": domain, "role": role, "checks": [extra]})
            success_by[(domain, role)] = {
                e: bool(record["completed"][e].get("success")) for e in done
            }
        teacher = success_by[(domain, "teacher")]
        student = success_by[(domain, "student")]
        common = sorted(set(teacher) & set(student))
        condition["pairs"] = {
            "status": "missing"
            if not common
            else "partial"
            if len(common) < expected
            else "complete",
            "common": len(common),
            "teacher_only": sorted(set(teacher) - set(student)),
            "student_only": sorted(set(student) - set(teacher)),
            "regressions": [e for e in common if teacher[e] and not student[e]],
            "recoveries": [e for e in common if student[e] and not teacher[e]],
            "both_success": sum(teacher[e] and student[e] for e in common),
            "both_failure": sum(not teacher[e] and not student[e] for e in common),
        }
        for episode in expected_ids:
            audit = _audit_pair(
                data[(domain, "teacher")]["completed"].get(episode),
                data[(domain, "student")]["completed"].get(episode),
                data[(domain, "teacher")]["npzs"].get(episode),
                data[(domain, "student")]["npzs"].get(episode),
            )
            audit["domain"] = domain
            report["audits"].append(audit)
        report["conditions"][domain] = condition
    # Cross-domain paired noise-prefix audit; no hashes.
    for role in ROLES:
        episodes = sorted({e for domain in DOMAINS for e in data[(domain, role)]["npzs"]})
        for episode in episodes:
            arrays = []
            present = []
            for domain in DOMAINS:
                path = data[(domain, role)]["npzs"].get(episode)
                if path is not None:
                    with np.load(path, allow_pickle=False) as archive:
                        arrays.append(np.asarray(archive["noise"]))
                    present.append(domain)
            if len(arrays) < 2:
                report["cross_domain_audits"].append(
                    {
                        "episode": episode,
                        "role": role,
                        "domains": present,
                        "status": "missing",
                        "check": "noise_prefix",
                    }
                )
                continue
            count = min(array.shape[0] for array in arrays)
            ok = all(np.array_equal(arrays[0][:count], array[:count]) for array in arrays[1:])
            report["cross_domain_audits"].append(
                {
                    "episode": episode,
                    "role": role,
                    "domains": present,
                    "status": "pass" if ok else "fail",
                    "check": "noise_prefix",
                }
            )

    # Cross-domain retention relative to Normal.
    report["cross_domain"] = {}
    for domain in DOMAINS[1:]:
        per_role: dict[str, Any] = {}
        for role in ROLES:
            normal = success_by[("Normal", role)]
            target = success_by[(domain, role)]
            common = sorted(set(normal) & set(target))
            per_role[role] = {
                "common": len(common),
                "losses": [e for e in common if normal[e] and not target[e]],
                "recoveries": [e for e in common if target[e] and not normal[e]],
                "missing": sorted((set(normal) | set(target)) - set(common)),
            }
        report["cross_domain"][domain] = per_role
    return report


def _probe(path: Path) -> tuple[int, int, float]:
    raw = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height:format=duration",
            "-of",
            "json",
            str(path),
        ],
        text=True,
    )
    payload = json.loads(raw)
    stream = payload["streams"][0]
    duration = float(payload.get("format", {}).get("duration", 0.0))
    return int(stream["width"]), int(stream["height"]), duration


def _video_meta(root: Path, domain: str, episode: str, role: str) -> dict[str, Any]:
    path = root / domain / role / f"{role}_{episode}.npz"
    if not path.is_file():
        return {"status": "missing"}
    with np.load(path, allow_pickle=False) as archive:
        success = bool(np.asarray(archive["success"]).reshape(-1)[0])
        steps = int(np.asarray(archive["steps"]).reshape(-1)[0])
        lighting = _decode_lighting(archive) or {}
    return {
        "status": "present",
        "success": success,
        "steps": steps,
        "ev": lighting.get("ev"),
        "domain": lighting.get("lighting_domain"),
    }


def make_videos(root: str | Path, report: dict[str, Any]) -> None:
    root = Path(root)
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError("ffmpeg/ffprobe are required for --videos")
    out = root / "comparison"
    out.mkdir(exist_ok=True)
    cards = []
    for domain in DOMAINS:
        for episode in report["expected_episode_ids"]:
            tpath = root / domain / "teacher" / f"teacher_{episode}.mp4"
            spath = root / domain / "student" / f"student_{episode}.mp4"
            tmeta = _video_meta(root, domain, episode, "teacher")
            smeta = _video_meta(root, domain, episode, "student")
            ev = tmeta.get("ev", smeta.get("ev"))
            label = (
                f"{domain} {episode} | EV={ev} | "
                f"teacher success={tmeta.get('success')} steps={tmeta.get('steps')} | "
                f"student success={smeta.get('success')} steps={smeta.get('steps')}"
            )
            video = out / f"{domain}_{episode}.mp4"
            item = {"domain": domain, "episode": episode, "video": None, "label": label}
            if tpath.is_file() and spath.is_file():
                tw, th, td = _probe(tpath)
                sw, sh, sd = _probe(spath)
                if (tw, th) != (sw, sh):
                    item["error"] = f"resolution mismatch {tw}x{th} vs {sw}x{sh}"
                else:
                    longest = max(td, sd)
                    pad_t = max(0.0, sd - td)
                    pad_s = max(0.0, td - sd)
                    filters = (
                        f"[0:v]tpad=stop_mode=clone:stop_duration={pad_t:.3f}[a];"
                        f"[1:v]tpad=stop_mode=clone:stop_duration={pad_s:.3f}[b];"
                        f"[a][b]hstack=inputs=2[v]"
                    )
                    subprocess.run(
                        [
                            "ffmpeg",
                            "-y",
                            "-i",
                            str(tpath),
                            "-i",
                            str(spath),
                            "-filter_complex",
                            filters,
                            "-map",
                            "[v]",
                            "-t",
                            f"{longest:.3f}",
                            "-c:v",
                            "libx264",
                            "-pix_fmt",
                            "yuv420p",
                            "-an",
                            str(video),
                        ],
                        check=True,
                        capture_output=True,
                    )
                    item["video"] = str(video.relative_to(root))
            else:
                item["error"] = "missing teacher or student video"
            cards.append(item)
            report["video_pairs"].append(item)
    rows = "\n".join(
        f"<h2>{item['domain']} / {item['episode']}</h2><p>{item['label']}</p>"
        + (
            f'<video controls width="960" src="{item["video"]}"></video>'
            if item.get("video")
            else f"<p>MISSING: {item.get('error')}</p>"
        )
        for item in cards
    )
    (root / "videos.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>LOWLIGHT paired videos</title>"
        f"<h1>LOWLIGHT paired videos</h1>{rows}\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--videos", action="store_true")
    args = parser.parse_args(argv)
    root = Path(args.root)
    report = build_report(root)
    report_path = root / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.videos:
        make_videos(root, report)
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(Path(args.root) / "report.json"),
                "conditions": {
                    d: {r: report["conditions"][d]["roles"][r]["completed"] for r in ROLES}
                    for d in DOMAINS
                },
                "audit_failures": sum(
                    1
                    for audit in report["audits"]
                    for check in audit["checks"]
                    if check.get("status") == "fail"
                )
                + sum(1 for audit in report["cross_domain_audits"] if audit["status"] == "fail"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
