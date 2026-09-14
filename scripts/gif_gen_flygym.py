from pathlib import Path

import chess
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

import fly_chess_inference as core
import fly_chess_inference_flygym as flygym_app
import gif_gen


MAX_FRAMES = 20
FRAME_MS = 700


def plot_image(trace):
    fig = core.activity_plot(trace)
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    image = Image.fromarray(rgba).convert("RGB")
    plt.close(fig)
    return image


def fly_image():
    frames = next(iter(flygym_app.renderer.frames.values()))
    if not frames:
        raise RuntimeError("FlyGym did not render any frames.")
    return Image.fromarray(frames[-1]).convert("RGB")


def combine(left, right, label):
    height = max(left.height, right.height)

    if left.height != height:
        width = round(left.width * height / left.height)
        left = left.resize((width, height))
    if right.height != height:
        width = round(right.width * height / right.height)
        right = right.resize((width, height))

    top = 55
    image = Image.new("RGB", (left.width + right.width, height + top), "#181818")
    image.paste(left, (0, top))
    image.paste(right, (left.width, top))

    draw = ImageDraw.Draw(image)
    draw.text((18, 14), label, font=gif_gen.font(28, True), fill="white")
    return image


def main():
    board = chess.Board()
    frames = []

    try:
        for ply, san in enumerate(gif_gen.SAN_MOVES[:MAX_FRAMES], 1):
            move = board.parse_san(san)
            board.push(move)

            trace = core.run_connectome(board, trace=True)
            trace["move"] = move
            trace["move_san"] = san

            flygym_app.render_fly(trace)

            who = "White" if ply % 2 else "Black"
            label = f"{(ply + 1) // 2}. {who}: {san}"
            frames.append(combine(plot_image(trace), fly_image(), label))

        output_file = Path(__file__).resolve().parent.parent / "chess_flygym_neurons.gif"
        durations = [FRAME_MS] * len(frames)
        durations[-1] = 1800
        frames[0].save(
            output_file,
            save_all=True,
            append_images=frames[1:],
            duration=durations,
            loop=0,
            disposal=2,
        )
        print(f"Wrote {output_file.name}")
    finally:
        if flygym_app.fly_sim is not None:
            flygym_app.fly_sim.close()


if __name__ == "__main__":
    main()
