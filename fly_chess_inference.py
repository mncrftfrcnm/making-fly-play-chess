from pathlib import Path
import html
import os
import time

import chess
import chess.svg
import gradio as gr
import joblib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.sparse as sparse


# Model path
model_file = Path(__file__).resolve().parent / "fly_chess_rl.joblib"
if not model_file.is_file():
    model_file = Path("/content/fly_chess_rl.joblib")

BOARD_FEATURES = 64 * 12 + 1 + 4 + 8 + 1
PROPAGATION_STEPS = 4
RANDOM_SEED = 6
START_FEN = chess.STARTING_FEN
MAX_PLIES = 200

# Only the strongest activity is drawn so the graph stays readable
TRACE_NEURONS = 55
TRACE_READOUTS = 10
TRACE_EDGES = 100

rng = np.random.default_rng(RANDOM_SEED)


# Load the saved model
saved = joblib.load(model_file)

value_weights = np.asarray(saved["value_weights"], dtype=np.float32)
matrix = sparse.csr_matrix(saved["connectome"], dtype=np.float32)
readout_indices = np.asarray(saved["readout_indices"], dtype=np.int64)
selected = np.asarray(
    saved.get("selected_neurons", np.arange(matrix.shape[0])),
    dtype=np.int64,
)

if "propagation_steps" in saved:
    PROPAGATION_STEPS = int(saved["propagation_steps"])

coo = matrix.tocoo()
edge_post = coo.row.astype(np.int64)
edge_pre = coo.col.astype(np.int64)
edge_weight = coo.data.astype(np.float32)
readout_set = set(readout_indices.tolist())

print(f"Loaded {model_file.name}")
print(f"Using {matrix.shape[0]:,} neurons and {matrix.nnz:,} fixed connections")
print(f"Using {len(readout_indices):,} readout neurons")


def encode_board(board):
    features = np.zeros(BOARD_FEATURES, dtype=np.float32)

    for square, piece in board.piece_map().items():
        color_offset = 0 if piece.color == chess.WHITE else 6
        channel = color_offset + piece.piece_type - 1
        features[square * 12 + channel] = 1.0

    cursor = 64 * 12
    features[cursor] = 1.0 if board.turn == chess.WHITE else -1.0
    cursor += 1

    features[cursor : cursor + 4] = (
        board.has_kingside_castling_rights(chess.WHITE),
        board.has_queenside_castling_rights(chess.WHITE),
        board.has_kingside_castling_rights(chess.BLACK),
        board.has_queenside_castling_rights(chess.BLACK),
    )
    cursor += 4

    if board.ep_square is not None:
        features[cursor + chess.square_file(board.ep_square)] = 1.0

    cursor += 8
    features[cursor] = min(board.halfmove_clock, 100) / 100.0
    return features


def run_connectome(board, trace=False):
    state = np.zeros(matrix.shape[0], dtype=np.float32)
    drive = np.zeros_like(state)
    drive[:BOARD_FEATURES] = encode_board(board)
    states = []

    for _ in range(PROPAGATION_STEPS):
        proposal = np.tanh(1.25 * (matrix @ state) + 1.5 * drive)
        state = 0.35 * state + 0.65 * proposal

        if trace:
            states.append(state.copy())

    features = state[readout_indices].copy()
    length = np.linalg.norm(features)
    if length > 0:
        features = features / length

    raw_value = float(value_weights[:-1] @ features + value_weights[-1])
    value = float(np.tanh(raw_value))

    if not trace:
        return value

    return {
        "states": np.asarray(states, dtype=np.float32),
        "contribution": value_weights[:-1] * features,
        "value": value,
    }


def terminal_reward(board):
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return None
    if outcome.winner == chess.WHITE:
        return 1.0
    if outcome.winner == chess.BLACK:
        return -1.0
    return 0.0


def predict_value(board):
    return run_connectome(board)


def choose_move(board):
    moves = list(board.legal_moves)
    scores = []

    for move in moves:
        next_board = board.copy(stack=True)
        next_board.push(move)

        score = terminal_reward(next_board)
        if score is None:
            score = predict_value(next_board)

        scores.append(float(score))

    best_score = max(scores) if board.turn == chess.WHITE else min(scores)
    best_moves = [
        move for move, score in zip(moves, scores)
        if np.isclose(score, best_score)
    ]
    move = best_moves[int(rng.integers(len(best_moves)))]

    next_board = board.copy(stack=True)
    next_board.push(move)
    trace = run_connectome(next_board, trace=True)

    trace["move"] = move
    trace["move_san"] = board.san(move)
    trace["policy_score"] = best_score
    trace["moves"] = moves
    trace["scores"] = scores

    return move, trace


def neuron_name(index):
    return str(int(selected[index])) if index < len(selected) else str(int(index))


def blank_plot(text="Waiting for the fly to move"):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=14)
    ax.set_axis_off()
    fig.tight_layout()
    return fig


def activity_plot(trace):
    states = trace["states"]
    final_state = states[-1]
    activity = np.max(np.abs(states), axis=0)

    top_active = np.argsort(activity)[-TRACE_NEURONS:]

    contribution = trace["contribution"]
    top_readout_pos = np.argsort(np.abs(contribution))[-TRACE_READOUTS:]
    top_readouts = readout_indices[top_readout_pos]

    show_nodes = np.unique(
        np.concatenate((top_active, top_readouts))
    ).astype(np.int64)
    node_set = set(show_nodes.tolist())

    keep = np.isin(edge_pre, show_nodes) & np.isin(edge_post, show_nodes)
    pre = edge_pre[keep]
    post = edge_post[keep]
    weights = edge_weight[keep]

    signal = np.abs(weights * final_state[pre])
    if len(signal) > TRACE_EDGES:
        strongest = np.argsort(signal)[-TRACE_EDGES:]
        pre = pre[strongest]
        post = post[strongest]
        signal = signal[strongest]

    graph = nx.DiGraph()
    graph.add_nodes_from(show_nodes.tolist())

    for source, target, strength in zip(pre, post, signal):
        graph.add_edge(int(source), int(target), signal=float(strength))

    pos = nx.spring_layout(
        graph,
        seed=RANDOM_SEED,
        iterations=40,
        weight="signal",
    )

    fig, ax = plt.subplots(figsize=(9, 7))

    input_nodes = [node for node in graph.nodes if node < BOARD_FEATURES]
    readout_nodes = [node for node in graph.nodes if node in readout_set]
    reservoir_nodes = [
        node for node in graph.nodes
        if node >= BOARD_FEATURES and node not in readout_set
    ]

    max_activity = max(float(np.max(activity[show_nodes])), 1e-6)

    def node_sizes(nodes):
        return [
            70 + 600 * float(activity[node]) / max_activity
            for node in nodes
        ]

    def node_colors(nodes):
        return [float(final_state[node]) for node in nodes]

    if graph.number_of_edges() > 0:
        max_signal = max(
            max(graph[u][v]["signal"] for u, v in graph.edges),
            1e-6,
        )
        widths = [
            0.3 + 3.0 * graph[u][v]["signal"] / max_signal
            for u, v in graph.edges
        ]

        nx.draw_networkx_edges(
            graph,
            pos,
            width=widths,
            alpha=0.25,
            arrows=True,
            arrowsize=8,
            ax=ax,
        )

    drawn = None

    for nodes, shape, outline in (
        (reservoir_nodes, "o", 0.4),
        (input_nodes, "^", 0.8),
        (readout_nodes, "s", 1.8),
    ):
        if nodes:
            drawn = nx.draw_networkx_nodes(
                graph,
                pos,
                nodelist=nodes,
                node_size=node_sizes(nodes),
                node_color=node_colors(nodes),
                cmap="coolwarm",
                vmin=-1,
                vmax=1,
                node_shape=shape,
                edgecolors="black",
                linewidths=outline,
                ax=ax,
            )

    labels = {
        int(node): neuron_name(int(node))
        for node in top_readouts
        if int(node) in node_set
    }
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=7, ax=ax)

    if drawn is not None:
        fig.colorbar(
            drawn,
            ax=ax,
            fraction=0.035,
            pad=0.02,
            label="Final activation",
        )

    ax.set_title(
        f"Reservoir activity after {trace['move_san']}  |  "
        f"value {trace['value']:+.3f}"
    )
    ax.text(
        0.01,
        0.01,
        "triangle = board input   square = readout   circle = reservoir\n"
        "size = strongest activity across propagation   arrows = strongest signals",
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
    )
    ax.set_axis_off()
    fig.tight_layout()
    return fig


def activity_text(trace):
    states = trace["states"]
    activity = np.max(np.abs(states), axis=0)
    active = int(np.sum(activity > 0.01))

    contribution = trace["contribution"]
    top = np.argsort(np.abs(contribution))[-8:][::-1]

    lines = [
        f"### Fly chose **{html.escape(trace['move_san'])}**",
        f"Policy score: `{trace['policy_score']:+.3f}`  ",
        f"Network value: `{trace['value']:+.3f}`  ",
        f"Active neurons (> 0.01): **{active:,} / {matrix.shape[0]:,}**  ",
        f"Propagation steps: **{len(states)}**",
        "",
        "**Largest readout contributions**",
    ]

    for index in top:
        neuron = int(readout_indices[index])
        lines.append(
            f"- neuron `{neuron_name(neuron)}`: `{contribution[index]:+.4f}`"
        )

    return "\n".join(lines)


def candidate_text(board, trace):
    rows = []

    for move, score in zip(trace["moves"], trace["scores"]):
        rows.append((board.san(move), move.uci(), float(score)))

    rows.sort(key=lambda x: x[2], reverse=board.turn == chess.WHITE)

    lines = ["### Moves considered", ""]

    for san, uci, score in rows[:10]:
        marker = " ← chosen" if uci == trace["move"].uci() else ""
        lines.append(f"`{san:8}`  `{score:+.3f}`{marker}")

    return "  \n".join(lines)


def board_html(board, last_move=None, orientation=chess.WHITE):
    check_square = board.king(board.turn) if board.is_check() else None
    svg = chess.svg.board(
        board=board,
        orientation=orientation,
        lastmove=last_move,
        check=check_square,
        size=520,
        coordinates=True,
    )
    return '<div style="display:flex;justify-content:center;">' + svg + "</div>"


def game_result(board):
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return ""

    result = board.result(claim_draw=True)
    if outcome.winner == chess.WHITE:
        return f"White wins ({result})"
    if outcome.winner == chess.BLACK:
        return f"Black wins ({result})"
    return f"Draw ({result})"


def new_state(mode, human_color=None):
    return {
        "mode": mode,
        "moves": [],
        "san": [],
        "human_color": human_color,
    }


def state_board(state):
    board = chess.Board(START_FEN)
    for move in state["moves"]:
        board.push_uci(move)
    return board


def push_move(board, state, move):
    san = board.san(move)
    board.push(move)
    state["moves"].append(move.uci())
    state["san"].append(san)
    return san


def move_history(state):
    if not state["san"]:
        return "_No moves yet._"

    lines = []
    for i in range(0, len(state["san"]), 2):
        white = html.escape(state["san"][i])
        black = (
            html.escape(state["san"][i + 1])
            if i + 1 < len(state["san"])
            else ""
        )
        lines.append(f"**{i // 2 + 1}.** {white} &nbsp;&nbsp; {black}")

    return "  \n".join(lines)


def legal_moves(board, enabled):
    choices = []

    if enabled:
        for move in board.legal_moves:
            choices.append((f"{board.san(move)}   [{move.uci()}]", move.uci()))
        choices.sort(key=lambda x: x[0])

    return gr.Dropdown(
        choices=choices,
        value=None,
        label="Your legal move",
        interactive=enabled,
        filterable=True,
    )


def screen_values(state, board, status, trace=None, old_board=None, last_move=None):
    human_color = state.get("human_color")
    orientation = human_color if human_color is not None else chess.WHITE

    human_turn = (
        state["mode"] == "You vs Fly"
        and not board.is_game_over(claim_draw=True)
        and board.turn == human_color
    )

    if trace is None:
        plot = blank_plot()
        neural = "### Neural activity\nWaiting for the fly to move."
        candidates = "### Moves considered\nWaiting for the fly to move."
    else:
        plot = activity_plot(trace)
        neural = activity_text(trace)
        candidates = candidate_text(old_board, trace)

    return (
        state,
        board_html(board, last_move, orientation),
        status,
        move_history(state),
        legal_moves(board, human_turn),
        plot,
        neural,
        candidates,
    )


def start_game(mode, side, delay, max_plies):
    if mode == "You vs Fly":
        human_color = chess.WHITE if side == "White" else chess.BLACK
        state = new_state(mode, human_color)
        board = state_board(state)

        if human_color == chess.BLACK:
            old_board = board.copy(stack=True)
            move, trace = choose_move(board)
            push_move(board, state, move)

            status = "### Your turn."
            yield screen_values(
                state,
                board,
                status,
                trace,
                old_board,
                move,
            )
        else:
            yield screen_values(
                state,
                board,
                "### You are White — your turn.",
            )
        return

    state = new_state(mode)
    board = state_board(state)

    yield screen_values(
        state,
        board,
        "### White Fly to move",
    )

    for _ in range(int(max_plies)):
        if board.is_game_over(claim_draw=True):
            break

        moving_side = "White Fly" if board.turn == chess.WHITE else "Black Fly"
        old_board = board.copy(stack=True)
        move, trace = choose_move(board)
        san = push_move(board, state, move)

        if board.is_game_over(claim_draw=True):
            status = "### " + game_result(board)
        else:
            next_side = "White Fly" if board.turn == chess.WHITE else "Black Fly"
            status = (
                f"### {moving_side} played **{html.escape(san)}**  \n"
                f"{next_side} to move."
            )

        yield screen_values(
            state,
            board,
            status,
            trace,
            old_board,
            move,
        )

        if delay > 0:
            time.sleep(float(delay))

    if not board.is_game_over(claim_draw=True):
        yield screen_values(
            state,
            board,
            f"### Stopped after {len(state['moves'])} plies.",
            last_move=board.peek() if board.move_stack else None,
        )


def human_move(state, move_uci):
    if not state or state.get("mode") != "You vs Fly":
        board = chess.Board(START_FEN)
        state = new_state("You vs Fly", chess.WHITE)
        return screen_values(state, board, "### Start a You vs Fly game first.")

    board = state_board(state)
    human_color = state["human_color"]

    if board.is_game_over(claim_draw=True):
        return screen_values(state, board, "### " + game_result(board))

    if board.turn != human_color:
        return screen_values(state, board, "### It is the fly's turn.")

    if not move_uci:
        return screen_values(state, board, "### Choose a legal move.")

    move = chess.Move.from_uci(move_uci)
    if move not in board.legal_moves:
        return screen_values(state, board, "### That move is no longer legal.")

    push_move(board, state, move)

    if board.is_game_over(claim_draw=True):
        return screen_values(
            state,
            board,
            "### " + game_result(board),
            last_move=move,
        )

    old_board = board.copy(stack=True)
    fly_move, trace = choose_move(board)
    push_move(board, state, fly_move)

    if board.is_game_over(claim_draw=True):
        status = "### " + game_result(board)
    else:
        status = "### Your turn."

    return screen_values(
        state,
        board,
        status,
        trace,
        old_board,
        fly_move,
    )


CSS = """
.board svg {
    width: min(520px, 96vw) !important;
    height: auto !important;
}
"""

with gr.Blocks(title="Fly Chess Neural Activity", css=CSS) as demo:
    gr.Markdown(
        "# Fly Chess — inference and neural activity\n"
        f"Loaded **{html.escape(model_file.name)}** · "
        f"**{matrix.shape[0]:,} neurons** · "
        f"**{matrix.nnz:,} fixed connections**"
    )

    game_state = gr.State(new_state("You vs Fly", chess.WHITE))

    with gr.Row():
        mode = gr.Radio(
            ["You vs Fly", "Fly vs Fly"],
            value="You vs Fly",
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
            MAX_PLIES,
            value=80,
            step=10,
            label="Fly vs Fly max plies",
        )

    start_button = gr.Button("Start / New Game", variant="primary")

    with gr.Row():
        with gr.Column(scale=1):
            board_view = gr.HTML(
                board_html(chess.Board(START_FEN)),
                elem_classes="board",
            )
            status = gr.Markdown("### Choose a mode and start.")
            move_select = gr.Dropdown(
                choices=[],
                label="Your legal move",
                interactive=False,
                filterable=True,
            )
            play_button = gr.Button("Play Move")

        with gr.Column(scale=1):
            neural_plot = gr.Plot(
                value=blank_plot(),
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

    outputs = [
        game_state,
        board_view,
        status,
        history,
        move_select,
        neural_plot,
        neural_info,
        candidates,
    ]

    start_button.click(
        start_game,
        inputs=[mode, side, delay, max_plies],
        outputs=outputs,
    )

    play_button.click(
        human_move,
        inputs=[game_state, move_select],
        outputs=outputs,
    )


if __name__ == "__main__":
    in_colab = "COLAB_RELEASE_TAG" in os.environ or "COLAB_GPU" in os.environ
    demo.queue().launch(share=in_colab)