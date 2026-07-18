import numpy as np

from app.core.turing import TuringSimulator

CONTROLS = {
    "F1": 0.04,
    "F2": 0.08,
    "K1": 0.056,
    "K2": 0.074,
    "Du1": 0.7,
    "Du2": 0.7,
    "Dv1": 0.25,
    "Dv2": 0.25,
}


def test_seeded_live_simulations_start_deterministically():
    first = TuringSimulator(CONTROLS, shape=(8, 8), seed=123)
    second = TuringSimulator(CONTROLS, shape=(8, 8), seed=123)

    np.testing.assert_array_equal(first.U, second.U)
    np.testing.assert_array_equal(first.V, second.V)
    np.testing.assert_array_equal(first.step(), second.step())


def test_normalizing_a_collapsed_frame_is_safe():
    frame = TuringSimulator.img_norm(np.ones((4, 4)))

    assert frame.dtype == np.uint8
    assert not frame.any()
