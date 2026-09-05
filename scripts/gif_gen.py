from pathlib import Path

import chess
from PIL import Image, ImageDraw, ImageFont


MOVE_TEXT = """1. g4 d5
2. Bh3 a5
3. b4 Be6
4. Ba3 Qd7
5. Nc3 c6
6. Ne4 dxe4
7. b5 Qd4
8. Qb1 cxb5
9. Nf3 Bxa2
10. Nxd4 Nf6
11. Nb3 e6
12. Qxa2 Bxa3
13. Qb1 Ke7
14. Qa2 a4
15. Qb1 axb3
16. Qa2 bxa2
17. Rc1 a1=R
18. Rxa1 g6
19. Ra2 b6
20. Ra1 h5
21. Ra2 Nh7
22. d4 hxg4
23. Ra1 gxh3
24. Ra2 f6
25. Ra1 f5
26. Ra2 f4
27. Ra1 f3
28. Ra2 fxe2
29. Ra1 e3
30. Ra2 exf2+
31. Kxf2 Nf8
32. Ke1 Nh7
33. Ra1 e5
34. Rf1 Ke6
35. Rf3 exd4
36. Ra2 g5
37. Ra1 g4
38. Ra2 gxf3
39. Ra1 f2+
40. Kxf2 Nf8
41. Ke1 Rh7
42. Rb1 Rg7
43. Ra1 b4
44. Rb1 b3
45. Ra1 bxc2
46. Rb1 cxb1=Q+
47. Kxe2 b5
48. Kf3 Nc6
49. Ke2 Ne7
50. Kf3 b4"""

SAN_MOVES = [token for token in MOVE_TEXT.split() if not token.endswith(".")]
FILES = "abcdefgh"
PIECE = {
    "K": "♔",
    "Q": "♕",
    "R": "♖",
    "B": "♗",
    "N": "♘",
    "P": "♙",
    "k": "♚",
    "q": "♛",
    "r": "♜",
    "b": "♝",
    "n": "♞",
    "p": "♟",
}


def font(size, bold=False):
    paths = [
        (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            if bold
            else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        ),
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ]
    for path in paths:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def frame(board, label, last=None):
    square_size, top = 80, 86
    image = Image.new(
        "RGB", (8 * square_size, top + 8 * square_size + 34), "#181818"
    )
    draw = ImageDraw.Draw(image)
    light, dark = "#f0d9b5", "#b58863"
    highlight_light, highlight_dark = "#f6f669", "#d6d640"

    draw.text((18, 18), label, font=font(34, True), fill="white")
    for rank in range(8):
        for file in range(8):
            x = file * square_size
            y = top + (7 - rank) * square_size
            color = light if (file + rank) % 2 else dark
            square = chess.square(file, rank)
            if last and square in last:
                color = highlight_light if color == light else highlight_dark
            draw.rectangle((x, y, x + square_size, y + square_size), fill=color)

    piece_font = font(58)
    for square, piece in board.piece_map().items():
        file = chess.square_file(square)
        rank = chess.square_rank(square)
        x = file * square_size
        y = top + (7 - rank) * square_size
        glyph = PIECE[piece.symbol()]
        box = draw.textbbox((0, 0), glyph, font=piece_font)
        width, height = box[2] - box[0], box[3] - box[1]
        fill = "#fafafa" if piece.color == chess.WHITE else "#111111"
        stroke = "#111111" if piece.color == chess.WHITE else "#f5f5f5"
        draw.text(
            (x + (square_size - width) / 2, y + (square_size - height) / 2 - 4),
            glyph,
            font=piece_font,
            fill=fill,
            stroke_width=1,
            stroke_fill=stroke,
        )

    coordinate_font = font(18, True)
    for file in range(8):
        draw.text(
            (file * square_size + 5, top + 8 * square_size + 5),
            FILES[file],
            font=coordinate_font,
            fill="#dddddd",
        )
    return image


def main():
    board = chess.Board()
    frames = [frame(board, "Start position")]
    durations = [1000]

    for ply, san in enumerate(SAN_MOVES, 1):
        move = board.parse_san(san)
        who = "White" if board.turn == chess.WHITE else "Black"
        board.push(move)
        move_number = (ply + 1) // 2
        frames.append(
            frame(
                board,
                f"{move_number}. {who}: {san}",
                (move.from_square, move.to_square),
            )
        )
        durations.append(480)

    durations[-1] = 2200
    output_file = Path(__file__).resolve().parent.parent / "chess_selfplay.gif"
    frames[0].save(
        output_file,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        disposal=2,
    )
    print(f"Wrote {output_file.name}")


if __name__ == "__main__":
    main()
