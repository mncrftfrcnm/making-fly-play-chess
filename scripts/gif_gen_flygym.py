from pathlib import Path
import os

import chess
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageOps

import fly_chess_inference as core
import fly_chess_inference_flygym as flygym_app
import gif_gen


MAX_FRAMES = 20
FLY_FRAMES_PER_MOVE = 8
FRAME_MS = 60
PANEL_HEIGHT = 420
MOTION_THRESHOLD = 14


def plot_image(trace):
    fig = core.activity_plot(trace)
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    image = Image.fromarray(rgba).convert("RGB")
    plt.close(fig)
    return image


def render_fly_images(trace):
    flygym_app.setup_flygym()

    keyframes = flygym_app.fly_keyframes(trace["states"])
    neutral = flygym_app.steps.default_pose_by_dof_order(flygym_app.dof_order)
    flygym_app.renderer.reset()

    sim_steps = max(
        1,
        int(flygym_app.FLY_CLIP_SECONDS / flygym_app.fly_sim.timestep),
    )
    last_keyframe = len(keyframes) - 1

    for step in range(sim_steps):
        position = (
            0.0
            if sim_steps == 1
            else last_keyframe * step / (sim_steps - 1)
        )
        action = flygym_app.locomotion_action(
            joint_angles=neutral + flygym_app.interpolate(keyframes, position),
            adhesion_onoff=np.ones(6, dtype=bool),
        )
        flygym_app.apply_locomotion_action(
            flygym_app.fly_sim,
            flygym_app.fly.name,
            action,
        )
        flygym_app.fly_sim.step_with_profile()
        flygym_app.fly_sim.render_as_needed_with_profile()

    rendered = next(
        (frames for frames in flygym_app.renderer.frames.values() if frames),
        None,
    )
    if not rendered:
        raise RuntimeError("FlyGym did not render any frames.")

    count = min(FLY_FRAMES_PER_MOVE, len(rendered))
    indexes = np.linspace(0, len(rendered) - 1, count, dtype=int)
    return [
        Image.fromarray(rendered[index]).convert("RGB")
        for index in indexes
    ]


def resize_height(image, height=PANEL_HEIGHT):
    if image.height == height:
        return image

    width = round(image.width * height / image.height)
    return image.resize((width, height), Image.Resampling.LANCZOS)


def highlight_motion(previous, current):
    if previous is None:
        return current

    diff = ImageChops.difference(previous, current).convert("L")
    mask = diff.point(lambda value: 170 if value > MOTION_THRESHOLD else 0)

    overlay = Image.new("RGBA", current.size, (0, 255, 180, 0))
    overlay.putalpha(mask)
    return Image.alpha_composite(current.convert("RGBA"), overlay).convert("RGB")


def decorate_fly(image):
    image = ImageOps.expand(image, border=3, fill="#00d18f")
    draw = ImageDraw.Draw(image)
    draw.text(
        (12, 10),
        "Fly movement",
        font=gif_gen.font(22, True),
        fill="#00ffb3",
    )
    return image


def combine(board, neurons, fly):
    board = resize_height(board)
    neurons = resize_height(neurons)
    fly = resize_height(fly)

    image = Image.new(
        "RGB",
        (board.width + neurons.width + fly.width, PANEL_HEIGHT),
        "#181818",
    )
    image.paste(board, (0, 0))
    image.paste(neurons, (board.width, 0))
    image.paste(fly, (board.width + neurons.width, 0))
    return image


def main():
    board = chess.Board()
    frames = []
    previous_fly = None

    if os.name != "nt" and not os.environ.get("DISPLAY"):
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    try:
        for ply, san in enumerate(gif_gen.SAN_MOVES[:MAX_FRAMES], 1):
            move = board.parse_san(san)
            who = "White" if board.turn == chess.WHITE else "Black"
            board.push(move)

            trace = core.run_connectome(board, trace=True)
            trace["move"] = move
            trace["move_san"] = san

            label = f"{(ply + 1) // 2}. {who}: {san}"
            board_image = gif_gen.frame(
                board,
                label,
                (move.from_square, move.to_square),
            )
            neurons = plot_image(trace)

            for fly_raw in render_fly_images(trace):
                fly = highlight_motion(previous_fly, fly_raw)
                frames.append(combine(board_image, neurons, decorate_fly(fly)))
                previous_fly = fly_raw

        if not frames:
            raise RuntimeError("No GIF frames were generated.")

        output_file = (
            Path(__file__).resolve().parent.parent
            / "assets"
            / "chess_flygym_neurons.gif"
        )
        durations = [FRAME_MS] * len(frames)
        durations[-1] = 1600

        frames[0].save(
            output_file,
            save_all=True,
            append_images=frames[1:],
            duration=durations,
            loop=0,
            disposal=2,
            optimize=False,
        )
        print(f"Wrote {output_file.name} with {len(frames)} frames")
    finally:
        if flygym_app.fly_sim is not None:
            flygym_app.fly_sim.close()


if __name__ == "__main__":
    main()
