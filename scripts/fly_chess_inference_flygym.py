import html
import os
import sys
import tempfile
import uuid
from pathlib import Path

import chess
import gradio as gr
import numpy as np
from gradio_chessboard import Chessboard

import fly_chess_inference as core


FLY_CLIP_SECONDS = 0.6
FLY_VIDEO_FPS = 30
FLY_CAMERA_RES = (360, 480)
JOINT_MOVEMENT_RADIANS = 0.25

fly_sim = None
fly = None
renderer = None
steps = None
dof_order = None
locomotion_action = None
apply_locomotion_action = None

video_dir = Path(tempfile.gettempdir()) / "fly_chess_flygym"
video_dir.mkdir(parents=True, exist_ok=True)
video_files = []


def setup_flygym():
    global fly_sim, fly, renderer, steps, dof_order
    global locomotion_action, apply_locomotion_action

    if fly_sim is not None:
        return

    if sys.version_info < (3, 12):
        raise RuntimeError("FlyGym 2.1.0 requires Python 3.12 or newer.")

    if "google.colab" in sys.modules:
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    try:
        from flygym import Simulation
        from flygym.anatomy import ContactBodiesPreset
        from flygym.compose import FlatGroundWorld
        from flygym.utils.math import Rotation3D
        from flygym_demo.complex_terrain import (
            LocomotionAction,
            PreprogrammedSteps,
            apply_locomotion_action as apply_action,
            make_locomotion_fly,
        )
    except ImportError as exc:
        raise RuntimeError(
            'FlyGym is not installed. Run: pip install "flygym==2.1.0"'
        ) from exc

    fly = make_locomotion_fly(
        name="fly_chess_3d",
        add_adhesion=True,
        colorize=True,
    )
    camera = fly.add_tracking_camera(
        name="body_cam",
        pos_offset=(-0.5, -7.5, 0.0),
        rotation=Rotation3D("euler", (1.57, 0.0, 0.0)),
        fovy=35.0,
    )

    world = FlatGroundWorld()
    world.add_fly(
        fly,
        [0, 0, 0.8],
        Rotation3D("quat", [1, 0, 0, 0]),
        bodysegs_with_ground_contact=ContactBodiesPreset.TIBIA_TARSUS_ONLY,
        add_ground_contact_sensors=False,
    )

    fly_sim = Simulation(world)
    renderer = fly_sim.set_renderer(
        [camera],
        camera_res=FLY_CAMERA_RES,
        playback_speed=1.0,
        output_fps=FLY_VIDEO_FPS,
    )
    steps = PreprogrammedSteps()
    dof_order = fly.get_actuated_jointdofs_order("position")
    locomotion_action = LocomotionAction
    apply_locomotion_action = apply_action
    reset_flygym()


def reset_flygym():
    if fly_sim is None:
        return

    fly_sim.reset()
    action = locomotion_action(
        joint_angles=steps.default_pose_by_dof_order(dof_order),
        adhesion_onoff=np.ones(6, dtype=bool),
    )
    apply_locomotion_action(fly_sim, fly.name, action)
    fly_sim.warmup()
    renderer.reset()


def fly_keyframes(states):
    groups = np.array_split(
        np.arange(len(core.readout_indices)),
        len(dof_order),
    )
    keyframes = []

    for state in states:
        features = state[core.readout_indices].copy()
        length = np.linalg.norm(features)
        if length > 0:
            features /= length

        contribution = features * core.value_weights[:-1]
        joint_signal = np.array(
            [np.sum(contribution[group]) for group in groups],
            dtype=np.float32,
        )
        scale = max(float(np.max(np.abs(joint_signal))), 1e-8)
        keyframes.append(
            JOINT_MOVEMENT_RADIANS * np.tanh(joint_signal / scale)
        )

    return np.asarray(keyframes, dtype=np.float32)


def interpolate(keyframes, position):
    if len(keyframes) == 1:
        return keyframes[0]

    position = float(np.clip(position, 0.0, len(keyframes) - 1))
    left = int(np.floor(position))
    right = min(left + 1, len(keyframes) - 1)
    amount = position - left
    return (1.0 - amount) * keyframes[left] + amount * keyframes[right]


def render_fly(trace):
    setup_flygym()

    keyframes = fly_keyframes(trace["states"])
    neutral = steps.default_pose_by_dof_order(dof_order)
    renderer.reset()

    sim_steps = max(1, int(FLY_CLIP_SECONDS / fly_sim.timestep))
    last_keyframe = len(keyframes) - 1

    for step in range(sim_steps):
        position = 0.0 if sim_steps == 1 else last_keyframe * step / (sim_steps - 1)
        action = locomotion_action(
            joint_angles=neutral + interpolate(keyframes, position),
            adhesion_onoff=np.ones(6, dtype=bool),
        )
        apply_locomotion_action(fly_sim, fly.name, action)
        fly_sim.step_with_profile()
        fly_sim.render_as_needed_with_profile()

    path = video_dir / f"fly_{uuid.uuid4().hex[:8]}_{trace['move'].uci()}.mp4"
    renderer.save_video(path)
    video_files.append(path)

    while len(video_files) > 6:
        old_file = video_files.pop(0)
        try:
            old_file.unlink(missing_ok=True)
        except OSError:
            pass

    return str(path)


def trace_for_last_move(state):
    board = core.state_board(state)
    if not board.move_stack:
        return None

    trace = core.run_connectome(board, trace=True)
    trace["move"] = board.peek()
    return trace


def fly_output(state, enabled, moved=False):
    if not enabled:
        return (
            gr.update(value=None, visible=False),
            gr.update(visible=False),
        )

    if not moved:
        return (
            gr.update(value=None, visible=True),
            gr.update(value="### 3D fly\nWaiting for the fly to move.", visible=True),
        )

    try:
        trace = trace_for_last_move(state)
        video_path = render_fly(trace)
        text = (
            "### 3D fly\n"
            f"Movement from the **{len(core.readout_indices):,} readout neurons** "
            "used by the chess evaluator."
        )
        return (
            gr.update(value=video_path, visible=True),
            gr.update(value=text, visible=True),
        )
    except Exception as exc:
        return (
            gr.update(value=None, visible=True),
            gr.update(
                value=(
                    "### 3D fly unavailable\n"
                    f"`{html.escape(type(exc).__name__)}: {html.escape(str(exc))}`"
                ),
                visible=True,
            ),
        )


def show_fly_panel(enabled):
    return gr.update(visible=enabled), gr.update(visible=enabled)


def start_game(mode, side, delay, max_plies, top_moves, show_flygym):
    previous_moves = 0
    if show_flygym and fly_sim is not None:
        reset_flygym()

    for values in core.start_game(mode, side, delay, max_plies, top_moves):
        state = values[0]
        move_count = len(state["moves"])

        if mode == "Fly vs Fly":
            fly_moved = move_count > previous_moves
        else:
            human_color = state.get("human_color")
            fly_moved = (
                move_count > previous_moves
                and human_color == chess.BLACK
                and previous_moves == 0
            )

        previous_moves = move_count
        yield (*values, *fly_output(state, show_flygym, fly_moved))


def human_move(state, moved_fen, show_flygym):
    before = len(state.get("moves", [])) if state else 0
    values = core.human_move(state, moved_fen)
    new_state = values[0]
    fly_moved = len(new_state.get("moves", [])) >= before + 2
    return (*values, *fly_output(new_state, show_flygym, fly_moved))


CSS = core.CSS + """
.fly-video video {
    max-height: 520px;
    object-fit: contain;
}
"""

with gr.Blocks(title="Fly Chess Neural Activity", css=CSS) as demo:
    gr.Markdown(
        "# Fly Chess — inference and neural activity\n"
        f"Loaded **{html.escape(core.model_file.name)}** · "
        f"**{core.matrix.shape[0]:,} neurons** · "
        f"**{core.matrix.nnz:,} fixed connections**"
    )

    game_state = gr.State(core.new_state("Fly vs Fly"))

    with gr.Row():
        mode = gr.Radio(
            ["Fly vs Fly", "You vs Fly"],
            value="Fly vs Fly",
            label="Mode",
        )
        side = gr.Radio(["White", "Black"], value="White", label="Your side")
        delay = gr.Slider(
            0.0,
            2.0,
            value=0.35,
            step=0.05,
            label="Fly vs Fly delay",
        )
        max_plies = gr.Slider(
            10,
            core.MAX_PLIES,
            value=80,
            step=10,
            label="Fly vs Fly max plies",
        )
        top_moves = gr.Slider(
            1,
            10,
            value=core.DEFAULT_TOP_MOVES,
            step=1,
            label="Fly vs Fly top moves to choose from",
        )

    show_flygym = gr.Checkbox(
        value=False,
        label="Show 3D fly",
        info="Slower: renders a FlyGym simulation after each fly move.",
    )

    start_button = gr.Button("Start / New Game", variant="primary")

    with gr.Row():
        with gr.Column(scale=1):
            white_board = Chessboard(
                value=core.START_FEN,
                label="Drag a piece to make your move",
                interactive=True,
                game_mode=True,
                orientation="white",
                visible=False,
                elem_classes="board",
            )
            black_board = Chessboard(
                value=core.START_FEN,
                label="Drag a piece to make your move",
                interactive=True,
                game_mode=True,
                orientation="black",
                visible=False,
                elem_classes="board",
            )
            locked_board = gr.HTML(
                core.board_html(chess.Board(core.START_FEN)),
                visible=True,
                elem_classes="locked-board",
            )
            status = gr.Markdown("### Fly vs Fly selected — start a game.")

        with gr.Column(scale=1):
            neural_plot = gr.Plot(
                value=core.blank_plot(),
                label="Reservoir activity for the fly's chosen move",
            )
            neural_info = gr.Markdown(
                "### Neural activity\nWaiting for the fly to move."
            )

    with gr.Row():
        candidates = gr.Markdown(
            "### Moves considered\nWaiting for the fly to move."
        )
        history = gr.Markdown("_No moves yet._")

    with gr.Row():
        fly_video = gr.Video(
            value=None,
            label="3D fly movement",
            autoplay=True,
            loop=False,
            visible=False,
            elem_classes="fly-video",
        )
        fly_info = gr.Markdown(
            "### 3D fly\nWaiting for the fly to move.",
            visible=False,
        )

    outputs = [
        game_state,
        white_board,
        black_board,
        locked_board,
        status,
        history,
        neural_plot,
        neural_info,
        candidates,
        fly_video,
        fly_info,
    ]

    show_flygym.change(
        show_fly_panel,
        inputs=[show_flygym],
        outputs=[fly_video, fly_info],
    )

    start_button.click(
        start_game,
        inputs=[mode, side, delay, max_plies, top_moves, show_flygym],
        outputs=outputs,
    )

    white_board.move(
        human_move,
        inputs=[game_state, white_board, show_flygym],
        outputs=outputs,
        show_progress="minimal",
    )

    black_board.move(
        human_move,
        inputs=[game_state, black_board, show_flygym],
        outputs=outputs,
        show_progress="minimal",
    )


if __name__ == "__main__":
    in_colab = "COLAB_RELEASE_TAG" in os.environ or "COLAB_GPU" in os.environ

    try:
        demo.queue().launch(share=in_colab)
    finally:
        if fly_sim is not None:
            fly_sim.close()
