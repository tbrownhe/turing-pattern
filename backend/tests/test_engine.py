import numpy as np
import pytest

from app.core.engine import (
    SimulationError,
    TuringSimulator,
    endpoint_fields,
    laplacian,
    normalize_image,
)
from app.core.models import ControlValues, SimulationConfig
from app.core.turing import parameter_map, turing_pattern

CONTROLS = ControlValues(
    F1=0.04,
    F2=0.08,
    K1=0.056,
    K2=0.074,
    Du1=0.7,
    Du2=0.7,
    Dv1=0.25,
    Dv2=0.25,
)


def test_laplacian_wraps_across_periodic_edges() -> None:
    values = np.zeros((5, 5), dtype=np.float64)
    values[0, 0] = 1.0

    actual = laplacian(values)

    assert actual[0, 0] == pytest.approx(-1.0)
    assert actual[0, 1] == pytest.approx(0.2)
    assert actual[0, -1] == pytest.approx(0.2)
    assert actual[-1, 0] == pytest.approx(0.2)
    assert actual[-1, -1] == pytest.approx(0.05)
    assert actual.sum() == pytest.approx(0.0, abs=1e-15)


def test_endpoint_fields_are_broadcast_vectors() -> None:
    config = SimulationConfig(width=7, height=5, controls=CONTROLS, seed=4)

    fields = endpoint_fields(config, np.dtype("float32"))

    assert fields.feed.shape == (1, 7)
    assert fields.kill.shape == (1, 7)
    assert fields.diffusion_u.shape == (5, 1)
    assert fields.diffusion_v.shape == (5, 1)
    assert fields.nbytes == (7 + 7 + 5 + 5) * 4
    assert fields.feed[0, 0] == pytest.approx(CONTROLS.F1)
    assert fields.feed[0, -1] == pytest.approx(CONTROLS.F2)


def test_parameter_map_interpolates_and_validates_control_points() -> None:
    field = parameter_map(
        5,
        3,
        (0.0, 0.5, 1.0),
        (0.0, 1.0, 0.0),
        axis="x",
    )

    np.testing.assert_allclose(field, [[0.0, 0.5, 1.0, 0.5, 0.0]])
    with pytest.raises(ValueError, match="increase strictly"):
        parameter_map(5, 3, (0.0, 0.5, 0.5), (0.0, 1.0, 0.0))


def test_seeded_float32_output_matches_small_numerical_fixture() -> None:
    simulator = TuringSimulator(CONTROLS, shape=(3, 4), seed=123)

    frame = simulator.step(steps=3)

    np.testing.assert_array_equal(
        frame,
        [
            [30, 255, 35, 1],
            [159, 183, 162, 142],
            [3, 41, 11, 0],
        ],
    )
    np.testing.assert_allclose(
        simulator.V,
        [
            [0.05985905, 0.5088582, 0.07036748, 0.00324363],
            [0.31749186, 0.36500928, 0.3236433, 0.28417233],
            [0.00686482, 0.08180209, 0.02196746, -0.00070746],
        ],
        rtol=1e-6,
        atol=1e-7,
    )
    assert simulator.state.iteration == 3


def test_reset_updates_seed_and_recreates_the_seeded_state() -> None:
    simulator = TuringSimulator(CONTROLS, shape=(8, 8), seed=1)
    expected = TuringSimulator(CONTROLS, shape=(8, 8), seed=9)

    simulator.step(steps=2)
    simulator.reset(seed=9)

    assert simulator.seed == simulator.config.seed == 9
    assert simulator.state.iteration == 0
    np.testing.assert_array_equal(simulator.U, expected.U)
    np.testing.assert_array_equal(simulator.V, expected.V)

    simulator.step(steps=2)
    simulator.reset()
    np.testing.assert_array_equal(simulator.U, expected.U)
    np.testing.assert_array_equal(simulator.V, expected.V)


def test_non_finite_and_collapsed_normalization_are_explicit() -> None:
    with pytest.raises(SimulationError, match="non-finite"):
        normalize_image(np.array([[np.inf]], dtype=np.float32))

    np.testing.assert_array_equal(
        normalize_image(np.full((2, 2), 0.5, dtype=np.float32)),
        np.zeros((2, 2), dtype=np.uint8),
    )


def test_batch_engine_is_seeded_and_deterministic() -> None:
    first = turing_pattern(w=8, h=8, steps=4, upsample=1, seed=27)
    second = turing_pattern(w=8, h=8, steps=4, upsample=1, seed=27)

    np.testing.assert_array_equal(first, second)
