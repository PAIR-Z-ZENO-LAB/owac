import numpy as np
import torch

from owac.simulation.raw_student_closed_loop import (
    _burst,
    _packet,
    _wilson,
    paired_noise,
)


def _ring():
    return [
        {
            "agentview": np.full((256, 256, 3), i, np.uint8),
            "wrist": np.full((256, 256, 3), i + 10, np.uint8),
        }
        for i in range(6)
    ]


def test_six_frame_burst_and_packet_padding_causality():
    ring = _ring()
    burst = _burst(ring, "agentview")
    assert burst.shape == (6, 3, 256, 256)
    packet = _packet("ep", 10, ring, ("base_0_rgb", "left_wrist_0_rgb"), torch.device("cpu"))
    assert len(packet.frames) == 6 and packet.reset is True
    assert packet.frame_ids == (5, 6, 7, 8, 9, 10)
    assert packet.padding_mask == (False,) * 6
    assert packet.frames[0].values.shape == (1, 2, 3, 224, 224)
    assert packet.frames[-1].values.shape == packet.frames[0].values.shape
    later = _packet("ep", 15, ring, ("base_0_rgb", "left_wrist_0_rgb"), torch.device("cpu"))
    assert later.reset is False


def test_paired_noise_is_identical_for_same_episode_and_decision():
    a = paired_noise(101, "episode_1", 3)
    b = paired_noise(101, "episode_1", 3)
    c = paired_noise(101, "episode_1", 4)
    assert torch.equal(a, b) and a.shape == (1, 10, 32)
    assert not torch.equal(a, c)


def test_wilson_interval_is_denominator_correct():
    assert _wilson(0, 0) is None
    lo, hi = _wilson(16, 20)
    assert 0.0 <= lo < 16 / 20 < hi <= 1.0


def test_author_replay_loader_uses_file_without_old_hash_module(tmp_path):
    from owac.simulation.rawvla_teacher_controller import _load_author_replay_module

    script = tmp_path / "scripts/replay/replay_libero_train_rawvla_light_npz.py"
    script.parent.mkdir(parents=True)
    script.write_text("MARKER = 'direct-import-ok'\n", encoding="utf-8")
    module = _load_author_replay_module(tmp_path)
    assert module.MARKER == "direct-import-ok"


def test_physical_frame_ring_does_not_duplicate_frame_ten():
    from owac.simulation.raw_student_closed_loop import _push_raw

    ring = [{"frame_index": index} for index in range(11)]
    ring = _push_raw(ring, {"frame_index": 10})
    assert [item["frame_index"] for item in ring] == [5, 6, 7, 8, 9, 10]
    ring = _push_raw(ring, {"frame_index": 11})
    assert [item["frame_index"] for item in ring] == [6, 7, 8, 9, 10, 11]


def _fake_lighting_module():
    import types

    module = types.ModuleType("rawvla_bench.lighting")

    def parameters(domain, ev, sample_unit=0.0):
        return {"domain": domain, "ev": ev, "sample_unit": sample_unit, "noise_seed": 0}

    module.lighting_parameters_for_sample = parameters
    return module


def test_lighting_domain_sampling_and_normal_compatibility(monkeypatch):
    import sys

    from owac.simulation import rawvla_teacher_controller as controller

    monkeypatch.setitem(sys.modules, "rawvla_bench.lighting", _fake_lighting_module())
    monkeypatch.setattr(
        sys.modules["rawvla_bench.lighting"],
        "lighting_parameters_for_sample",
        sys.modules["rawvla_bench.lighting"].lighting_parameters_for_sample,
    )
    # Ensure the package namespace also resolves the injected module.
    package = sys.modules.get("rawvla_bench") or __import__("types").ModuleType("rawvla_bench")
    package.lighting = sys.modules["rawvla_bench.lighting"]
    monkeypatch.setitem(sys.modules, "rawvla_bench", package)

    from pathlib import Path

    adapter = controller.LIBEROEnvAdapter(
        author_root=Path("/tmp/a"),
        libero_config=Path("/tmp/c.yaml"),
        env_factory=lambda *_: None,
        lighting_domain="Normal",
    )
    normal = adapter._normal_lighting(101100)
    rng = np.random.default_rng(101100)
    ev = float(rng.uniform(-0.5, 0.5))
    unit = float(rng.random())
    assert normal["ev"] == ev
    assert normal["sample_unit"] == unit
    assert normal["noise_seed"] == (101100 ^ 0x5A17)
    assert normal["lighting_domain"] == "Normal"

    for domain, bounds in (("Low", (-4.0, -2.5)), ("ExtremeLow", (-5.5, -4.0))):
        adapter.lighting_domain = domain
        lighting = adapter._normal_lighting(101100)
        assert bounds[0] <= lighting["ev"] <= bounds[1]
        assert 0.0 <= lighting["sample_unit"] < 1.0
        assert lighting["lighting_domain"] == domain
        assert lighting["noise_seed"] == (101100 ^ 0x5A17)


def test_role_isolation_student_does_not_build_teacher_and_teacher_does_not_build_student(
    monkeypatch,
):
    from types import SimpleNamespace

    from owac.simulation import raw_student_closed_loop as module

    class Module:
        def eval(self):
            return self

        def parameters(self):
            return []

    class Student(Module):
        def load_state_dict(self, state, strict=True):
            assert state == {} and strict is True

        def __call__(self, *args):
            raise AssertionError("not called during construction")

    calls = {"teacher": 0, "student": 0, "load": 0}
    monkeypatch.setattr(module, "_build_bridge", lambda *a, **k: (object(), object()))
    monkeypatch.setattr(module, "build_all_bench_student", lambda *a, **k: Student())

    def teacher_factory(*a, **k):
        calls["teacher"] += 1
        return SimpleNamespace(frontend=Module(), vision_tower=Module(), projector=Module())

    monkeypatch.setattr(module, "build_vision_teacher", teacher_factory)
    monkeypatch.setattr(module.torch, "load", lambda *a, **k: {"model": {}})
    monkeypatch.setattr(module, "normalize_checkpoint_state", lambda payload: {})
    bridge, stats, teacher, student = module.build_role_models(
        {"assets": {}}, "ckpt.pt", device="cpu", role="student"
    )
    assert teacher is None and isinstance(student, Student)
    assert calls["teacher"] == 0
    calls["teacher"] = 0
    monkeypatch.setattr(
        module,
        "build_all_bench_student",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("student loaded")),
    )
    bridge, stats, teacher, student = module.build_role_models(
        {"assets": {}}, None, device="cpu", role="teacher"
    )
    assert student is None and teacher is not None
    assert calls["teacher"] == 1


def test_run_role_only_and_both_real_branches(tmp_path, monkeypatch):
    import sys
    import types
    from types import SimpleNamespace

    from owac.simulation import raw_student_closed_loop as module

    class Env:
        def reset(self):
            return {"raw": {}, "robot_state": np.zeros(8, np.float32), "native": {}}

        def step(self, action):
            return {"raw": {}, "robot_state": np.zeros(8, np.float32), "native": {}}, 0.0, True, {}

        def check_success(self):
            return True

        def close(self):
            pass

    class Adapter:
        def __init__(self, **kwargs):
            self.lighting = {"lighting_domain": kwargs["lighting_domain"], "ev": 0.0}

        def bind(self, entry, states):
            return Env()

    class Module:
        def eval(self):
            return self

        def parameters(self):
            return []

    class Student(Module):
        def load_state_dict(self, state, strict=True):
            pass

    counters = {"teacher_build": 0, "student_build": 0, "load": 0}

    def teacher_builder(*a, **k):
        counters["teacher_build"] += 1
        return SimpleNamespace(frontend=Module(), vision_tower=Module(), projector=Module())

    def student_builder(*a, **k):
        counters["student_build"] += 1
        return Student()

    def load(*a, **k):
        counters["load"] += 1
        return {"model": {}}

    class Bridge:
        def predict_action_from_visual_tokens(self, *args, **kwargs):
            return {"normalized_actions": np.zeros((1, 10, 32), np.float32)}

    class FakeTeacherTokens:
        constructed = 0
        last = None

        def __init__(self, *a):
            FakeTeacherTokens.constructed += 1
            FakeTeacherTokens.last = self
            self.states = {}
            self.thetas = {}
            self.calls = 0

        def tokens(self, ring):
            self.calls += 1
            return SimpleNamespace(values=torch.zeros(1, 1, 1), camera_ids=("a",), mask=None)

    class FakeStudentTokens:
        constructed = 0
        last = None

        def __init__(self, *a):
            FakeStudentTokens.constructed += 1
            FakeStudentTokens.last = self
            self.state = None
            self.calls = 0

        def tokens(self, episode_id, endpoint, ring):
            self.calls += 1
            return SimpleNamespace(values=torch.zeros(1, 1, 1), camera_ids=("a",), mask=None)

    class Writer:
        def append_data(self, frame):
            pass

        def close(self):
            pass

    fake_io = types.ModuleType("imageio.v2")
    fake_io.get_writer = lambda *a, **k: Writer()
    monkeypatch.setitem(sys.modules, "imageio.v2", fake_io)
    monkeypatch.setitem(sys.modules, "imageio", types.ModuleType("imageio"))
    monkeypatch.setattr(module, "LIBEROEnvAdapter", Adapter)
    monkeypatch.setattr(
        module,
        "load_test_states",
        lambda *a: ([{"id": "ep", "seed": 7, "array_index": 0}], np.zeros((1, 79), np.float32)),
    )
    monkeypatch.setattr(
        module,
        "_build_bridge",
        lambda *a, **k: (Bridge(), SimpleNamespace(state={"q01": [0.0] * 8, "q99": [1.0] * 8})),
    )
    monkeypatch.setattr(module, "build_vision_teacher", teacher_builder)
    monkeypatch.setattr(module, "build_all_bench_student", student_builder)
    monkeypatch.setattr(module.torch, "load", load)
    monkeypatch.setattr(module, "normalize_checkpoint_state", lambda payload: {})
    monkeypatch.setattr(module, "_postprocess", lambda *a: np.zeros((10, 7), np.float32))
    monkeypatch.setattr(
        module,
        "_raw_obs",
        lambda observation, cfg, **kw: {
            "agentview": np.zeros((256, 256, 3), np.uint8),
            "wrist": np.zeros((256, 256, 3), np.uint8),
            "frame_index": kw["frame_index"],
        },
    )
    monkeypatch.setattr(module, "_TeacherTokenController", FakeTeacherTokens)
    monkeypatch.setattr(module, "_StudentTokenController", FakeStudentTokens)
    from owac.backends import rawvla_language

    monkeypatch.setattr(
        rawvla_language,
        "prepare_language_state",
        lambda *a, **k: (
            torch.zeros(1, 2, dtype=torch.long),
            torch.ones(1, 2, dtype=torch.bool),
            torch.zeros(1, 32),
        ),
    )

    for role, expect_teacher, expect_student in (
        ("teacher", True, False),
        ("student", False, True),
        ("both", True, True),
    ):
        FakeTeacherTokens.constructed = 0
        FakeStudentTokens.constructed = 0
        FakeTeacherTokens.last = None
        FakeStudentTokens.last = None
        before = dict(counters)
        summary = module.run(
            {"assets": {"author_root": "/x"}},
            tmp_path / "catalog.json",
            "ckpt.pt",
            tmp_path / role,
            device="cpu",
            max_episodes=1,
            role=role,
        )
        assert (counters["teacher_build"] > before["teacher_build"]) is expect_teacher
        assert (counters["student_build"] > before["student_build"]) is expect_student
        assert FakeTeacherTokens.constructed == (1 if expect_teacher else 0)
        assert FakeStudentTokens.constructed == (1 if expect_student else 0)
        if expect_teacher:
            assert FakeTeacherTokens.last.calls == 1
            assert FakeTeacherTokens.last.states == {"agentview": None, "wrist": None}
        else:
            assert FakeTeacherTokens.last is None
        if expect_student:
            assert FakeStudentTokens.last.calls == 1
        else:
            assert FakeStudentTokens.last is None
        assert summary["role"] == role
        assert summary["paired"]["status"] == ("not_requested" if role != "both" else "complete")
