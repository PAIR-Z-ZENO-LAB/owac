import json

import numpy as np
import pytest

from owac.simulation.rgb_eval import decode_actions


def test_native_action_transport_preserves_float64_before_rotation_conversion():
    native = np.array(json.loads("[[0.12345678912345678, 0.999999999999]]"))
    decoded = decode_actions({"actions": native.tolist(), "dtype": native.dtype.str})
    assert decoded.dtype == native.dtype
    np.testing.assert_array_equal(decoded, native)
    assert not np.array_equal(decoded, native.astype(np.float32))


def test_native_action_transport_keeps_corp_float32():
    native = np.array([[0.123456789]], dtype=np.float32)
    decoded = decode_actions({"actions": native.tolist(), "dtype": native.dtype.str})
    assert decoded.dtype == native.dtype
    np.testing.assert_array_equal(decoded, native)


def test_new_checkpoint_evaluation_cannot_use_an_old_server():
    response = {"actions": [[0.1]], "dtype": "<f4", "checkpoint_sha256": "old"}
    with pytest.raises(ValueError, match="checkpoint"):
        decode_actions(response, checkpoint_sha256="new")
    response["checkpoint_sha256"] = "new"
    np.testing.assert_array_equal(
        decode_actions(response, checkpoint_sha256="new"), np.array([[0.1]], dtype=np.float32)
    )
