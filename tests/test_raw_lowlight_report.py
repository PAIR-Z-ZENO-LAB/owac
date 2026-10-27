import json
from pathlib import Path

import numpy as np

from owac.runtime.raw_lowlight_report import DOMAINS, build_report


def _make(
    root: Path,
    domain: str,
    role: str,
    completed: dict[str, bool],
    errors: list[str] | None = None,
    bad_executed: str | None = None,
) -> None:
    path = root / domain / role
    path.mkdir(parents=True)
    rows = []
    for episode, success in completed.items():
        rows.append(
            {
                "role": role,
                "episode_id": episode,
                "status": "completed",
                "seed": int(episode[-2:]),
                "success": success,
                "steps": 6,
            }
        )
        physical = np.arange(2 * 10 * 7, dtype=np.float32).reshape(2, 10, 7) + (
            1 if role == "teacher" else 2
        )
        executed = physical[:, :5].reshape(-1, 7)[:6].copy()
        if episode == bad_executed:
            executed[0, 0] = 999.0
        lighting = {
            "lighting_domain": domain,
            "ev": -1.0,
            "sample_unit": 0.25,
            "noise_seed": 123 ^ 0x5A17,
        }
        np.savez(
            path / f"{role}_{episode}.npz",
            normalized=np.zeros((2, 10, 32), np.float32),
            physical=physical,
            executed=executed,
            state=np.zeros((2, 8), np.float32),
            state_raw=np.zeros((2, 8), np.float32),
            noise=np.arange(2 * 10 * 32, dtype=np.float32).reshape(2, 10, 32),
            first_raw=np.zeros((6, 2, 8, 8, 3), np.uint8),
            initial_state=np.zeros(79, np.float64),
            initial_state_extra=np.asarray([0]),
            lighting_json=np.asarray(json.dumps(lighting)),
            episode_id=np.asarray([episode]),
            seed=np.asarray([int(episode[-2:])]),
            success=np.asarray([success]),
            steps=np.asarray([6]),
        )
    for episode in errors or []:
        rows.append(
            {"role": role, "episode_id": episode, "status": "error", "error": "environment_error"}
        )
    (path / "attempts.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _root(tmp_path: Path, *, bad_executed: str | None = None) -> Path:
    root = tmp_path / "run"
    root.mkdir()
    catalog = {
        "schema": "owac.raw-teacher.initial-states.v1",
        "episodes": [
            {"id": f"e{i:02d}", "split": "test", "seed": i, "array_index": i} for i in range(20)
        ],
    }
    (root / "catalog.json").write_text(json.dumps(catalog), encoding="utf-8")
    for domain in DOMAINS:
        _make(
            root,
            domain,
            "teacher",
            {"e00": True, "e01": True},
            errors=["e02"] if domain == "ExtremeLow" else None,
        )
        student_done = {"e00": True} if domain == "Low" else {"e00": True, "e01": True}
        _make(
            root,
            domain,
            "student",
            student_done,
            errors=["e02"] if domain == "ExtremeLow" else None,
            bad_executed=bad_executed if domain == "Normal" else None,
        )
    return root


def test_report_missing_error_denominators_and_pairs(tmp_path):
    report = build_report(_root(tmp_path))
    assert report["conditions"]["Normal"]["roles"]["teacher"]["success"] == 2
    assert report["conditions"]["Normal"]["roles"]["teacher"]["completed"] == 2
    assert report["conditions"]["Normal"]["roles"]["teacher"]["missing"] == 18
    assert report["conditions"]["Normal"]["pairs"]["common"] == 2
    assert report["conditions"]["Normal"]["pairs"]["regressions"] == []
    assert report["conditions"]["Low"]["pairs"]["regressions"] == []
    assert report["conditions"]["Low"]["pairs"]["teacher_only"] == ["e01"]
    assert report["conditions"]["ExtremeLow"]["roles"]["teacher"]["errors"] == 1
    assert "e02" in report["conditions"]["ExtremeLow"]["roles"]["teacher"]["error_ids"]
    assert report["conditions"]["ExtremeLow"]["pairs"]["common"] == 2
    # Missing pairs are not fabricated as failures.
    assert report["conditions"]["Low"]["pairs"]["both_failure"] == 0


def test_report_detects_real_npz_inconsistency(tmp_path):
    report = build_report(_root(tmp_path, bad_executed="e00"))
    failures = [
        (audit["domain"], check["check"])
        for audit in report["audits"]
        for check in audit["checks"]
        if check["status"] == "fail"
    ]
    assert ("Normal", "executed_matches_physical_student") in failures


def test_video_padding_clones_short_teacher_last_frame(tmp_path):
    import subprocess

    from owac.runtime.raw_lowlight_report import make_videos

    root = tmp_path / "videos"
    for role, color, frames in (("teacher", "red", 3), ("student", "blue", 5)):
        path = root / "Normal" / role
        path.mkdir(parents=True)
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                f"color=c={color}:s=16x16:r=10",
                "-frames:v",
                str(frames),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(path / f"{role}_e00.mp4"),
            ],
            check=True,
            capture_output=True,
        )
        np.savez(
            path / f"{role}_e00.npz",
            success=np.asarray([True]),
            steps=np.asarray([4]),
            physical=np.zeros((1, 10, 7), np.float32),
            executed=np.zeros((5, 7), np.float32),
            lighting_json=np.asarray(json.dumps({"ev": -3.0, "lighting_domain": "Normal"})),
            episode_id=np.asarray(["e00"]),
        )
    report = {
        "expected_episode_ids": ["e00"],
        "conditions": {
            d: {"roles": {r: {"completed": 0, "denominator": 20} for r in ("teacher", "student")}}
            for d in DOMAINS
        },
        "video_pairs": [],
    }
    make_videos(root, report)
    video = root / "comparison" / "Normal_e00.mp4"
    count = subprocess.check_output(
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
            str(video),
        ],
        text=True,
    ).strip()
    assert int(count.splitlines()[0]) == 5
    last = subprocess.check_output(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(video),
            "-vf",
            "select=eq(n\\,4)",
            "-vsync",
            "0",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ]
    )
    frame = np.frombuffer(last, np.uint8).reshape(16, 32, 3)
    left = frame[:, :16].mean(axis=(0, 1))
    right = frame[:, 16:].mean(axis=(0, 1))
    assert left[0] > 150 and left[2] < 100
    assert right[2] > 150 and right[0] < 100


def test_duplicate_and_unexpected_ids_are_reported_not_overwritten(tmp_path):
    root = _root(tmp_path)
    ledger = root / "Normal" / "teacher" / "attempts.jsonl"
    with ledger.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {
                    "role": "teacher",
                    "episode_id": "e00",
                    "status": "completed",
                    "success": False,
                    "steps": 6,
                }
            )
            + "\n"
        )
        stream.write(
            json.dumps(
                {
                    "role": "teacher",
                    "episode_id": "e99",
                    "status": "completed",
                    "success": True,
                    "steps": 6,
                }
            )
            + "\n"
        )
    report = build_report(root)
    checks = [
        check["check"]
        for audit in report["audits"]
        if audit["domain"] == "Normal"
        for check in audit["checks"]
    ]
    assert "duplicate_completed" in checks
    assert "unexpected_episode_id" in checks
    assert report["conditions"]["Normal"]["roles"]["teacher"]["success"] == 2


def test_video_frame_audit_matches_steps_plus_eleven(tmp_path):
    import subprocess

    from owac.runtime.raw_lowlight_report import _audit_role

    path = tmp_path / "teacher_e00.npz"
    video = tmp_path / "teacher_e00.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=green:s=16x16:r=10",
            "-frames:v",
            "15",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],
        check=True,
        capture_output=True,
    )
    physical = np.zeros((1, 10, 7), np.float32)
    np.savez(
        path,
        physical=physical,
        executed=physical[:, :5].reshape(-1, 7),
        success=np.asarray([True]),
        steps=np.asarray([4]),
        episode_id=np.asarray(["e00"]),
    )
    checks = _audit_role("teacher", {"success": True, "steps": 4}, path)
    assert {check["check"]: check["status"] for check in checks}[
        "video_frames_match_steps_teacher"
    ] == "pass"
