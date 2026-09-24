# pylint: disable=wrong-import-position,import-outside-toplevel,import-error

from pathlib import Path
import sys

import chess
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import fly_chess_inference as core
import fly_chess_inference_flygym as flygym_app
import gif_gen


def test_mujoco_import():
    import mujoco

    assert mujoco.__version__


def test_chess_imports():
    assert chess.Board is not None
    assert chess.Move is not None


def test_flygym_import():
    import flygym

    assert flygym is not None


def test_flygym_does_not_start_on_import():
    assert flygym_app.fly_sim is None


def test_model_shapes():
    assert core.matrix.shape[0] == len(core.selected)
    assert len(core.readout_indices) == len(core.value_weights) - 1
    assert np.all(core.readout_indices >= 0)
    assert np.all(core.readout_indices < core.matrix.shape[0])


def test_gradio():
    import gradio as gr

    def greet(name):
        return f"hello, {name}!"

    demo = gr.Interface(fn=greet, inputs="text", outputs="text")

    assert isinstance(demo, gr.Interface)
    assert greet("fly") == "hello, fly!"


def test_board_encoding():
    features = core.encode_board(chess.Board())

    assert features.shape == (core.BOARD_FEATURES,)
    assert np.all(np.isfinite(features))


def test_connectome_trace():
    trace = core.run_connectome(chess.Board(), trace=True)

    assert trace["states"].shape == (
        core.propagation_steps,
        core.matrix.shape[0],
    )
    assert trace["contribution"].shape == (len(core.readout_indices),)
    assert np.all(np.isfinite(trace["states"]))
    assert np.isfinite(trace["value"])
    assert -1.0 <= trace["value"] <= 1.0


def test_interpolate():
    keyframes = np.array(
        [
            [0.0, 2.0],
            [2.0, 4.0],
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(
        flygym_app.interpolate(keyframes, -1),
        [0.0, 2.0],
    )
    np.testing.assert_allclose(
        flygym_app.interpolate(keyframes, 0.5),
        [1.0, 3.0],
    )
    np.testing.assert_allclose(
        flygym_app.interpolate(keyframes, 2),
        [2.0, 4.0],
    )


def test_fly_keyframes(monkeypatch):
    trace = core.run_connectome(chess.Board(), trace=True)
    monkeypatch.setattr(flygym_app, "dof_order", list(range(42)))

    keyframes = flygym_app.fly_keyframes(trace["states"])

    assert keyframes.shape == (core.propagation_steps, 42)
    assert np.all(np.isfinite(keyframes))
    assert np.max(np.abs(keyframes)) <= flygym_app.JOINT_MOVEMENT_RADIANS + 1e-6


def test_trace_for_last_move():
    state = core.new_state("Fly vs Fly")
    assert flygym_app.trace_for_last_move(state) is None

    state["moves"].append("e2e4")
    trace = flygym_app.trace_for_last_move(state)

    assert trace is not None
    assert trace["move"] == chess.Move.from_uci("e2e4")
    assert trace["states"].shape[0] == core.propagation_steps


def test_PIL_and_gif_gen():
    from PIL import Image

    image = gif_gen.frame(chess.Board(), "Test frame")
    rendered_font = gif_gen.font(18)

    assert isinstance(image, Image.Image)
    assert image.mode == "RGB"
    assert image.size == (640, 760)
    assert hasattr(rendered_font, "getbbox")
