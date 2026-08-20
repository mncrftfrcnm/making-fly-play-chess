
from pathlib import Path

import chess
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sparse
from scipy.sparse.csgraph import breadth_first_order


# Model paths
model_folder = Path(__file__).resolve().parent / "Drosophila_brain_model"
neuron_file = model_folder / "Completeness_783.csv"
connections_file = model_folder / "Connectivity_783.parquet"
save_model_file = Path(__file__).resolve().parent / "fly_chess_rl.joblib"

# Hyperparameters
SELF_PLAY_GAMES = 10000
MAX_PLIES = 300
OPENING_TEST_PLIES = 200
READOUT_NEURONS = 1024
PROPAGATION_STEPS = 8
LEARNING_RATE = 0.002
DISCOUNT = 0.9953
EPSILON_START = 0.40
EPSILON_END = 0.02
MATERIAL_SHAPING = 0.08
RANDOM_SEED = 67
START_FEN = chess.STARTING_FEN

BOARD_FEATURES = 64 * 12 + 1 + 4 + 8 + 1


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


# Load the fixed reservoir
neurons = pd.read_csv(neuron_file, index_col=0)
RESERVOIR_NEURONS = 16384 # can be len(neurons), but it takes a lot of time to train


connections = pd.read_parquet(connections_file)

pre = connections["Presynaptic_Index"].to_numpy(dtype=np.int64)
post = connections["Postsynaptic_Index"].to_numpy(dtype=np.int64)
weight = connections["Excitatory x Connectivity"].to_numpy(dtype=np.float32)
neuron_count = len(neurons)

valid = (
    (pre >= 0)
    & (post >= 0)
    & (pre < neuron_count)
    & (post < neuron_count)
    & np.isfinite(weight)
    & (weight != 0)
)
pre, post, weight = pre[valid], post[valid], weight[valid]

degree = np.bincount(pre, minlength=neuron_count)
degree += np.bincount(post, minlength=neuron_count)
seed_neuron = int(np.argmax(degree))

structure = sparse.csr_matrix(
    (np.ones(len(pre), dtype=np.uint8), (post, pre)),
    shape=(neuron_count, neuron_count),
)
order = breadth_first_order(
    structure, seed_neuron, directed=False, return_predecessors=False
)
selected = np.asarray(order[:RESERVOIR_NEURONS], dtype=np.int64)

local_index = np.full(neuron_count, -1, dtype=np.int64)
local_index[selected] = np.arange(len(selected))
keep = (local_index[pre] >= 0) & (local_index[post] >= 0)

matrix = sparse.csr_matrix(
    (weight[keep], (local_index[post[keep]], local_index[pre[keep]])),
    shape=(len(selected), len(selected)),
    dtype=np.float32,
)
matrix.sum_duplicates()

row_total = np.asarray(abs(matrix).sum(axis=1)).ravel()
row_total[row_total < 1.0] = 1.0
matrix = (sparse.diags(1.0 / row_total) @ matrix).tocsr()

rng = np.random.default_rng(RANDOM_SEED)
readout_pool = np.arange(BOARD_FEATURES, len(selected))
readout_indices = np.sort(
    rng.choice(readout_pool, size=READOUT_NEURONS, replace=False)
)

print(f"{neuron_count:,} neurons were loaded")
print(f"using {len(selected):,} neurons and {matrix.nnz:,} fixed connections")


def connectome_features(board):
    state = np.zeros(len(selected), dtype=np.float32)
    drive = np.zeros_like(state)
    drive[:BOARD_FEATURES] = encode_board(board)

    for _ in range(PROPAGATION_STEPS):
        proposal = np.tanh(1.25 * (matrix @ state) + 1.5 * drive)
        state = 0.35 * state + 0.65 * proposal

    features = state[readout_indices]
    length = np.linalg.norm(features)
    if length > 0:
        features = features / length

    return np.append(features, 1.0).astype(np.float32)


def material_score(board):
    values = {
        chess.PAWN: 1.0,
        chess.KNIGHT: 3.2,
        chess.BISHOP: 3.3,
        chess.ROOK: 5.0,
        chess.QUEEN: 9.0,
    }
    return sum(
        value
        * (
            len(board.pieces(piece, chess.WHITE))
            - len(board.pieces(piece, chess.BLACK))
        )
        for piece, value in values.items()
    )


def terminal_reward(board):
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return None
    if outcome.winner == chess.WHITE:
        return 1.0
    if outcome.winner == chess.BLACK:
        return -1.0
    return 0.0


def predict_value(board, value_weights):
    features = connectome_features(board)
    return float(np.tanh(value_weights @ features))


def choose_move(board, value_weights, epsilon):
    moves = list(board.legal_moves)

    if rng.random() < epsilon:
        return moves[int(rng.integers(len(moves)))]

    scores = []
    for move in moves:
        next_board = board.copy(stack=True)
        next_board.push(move)
        score = terminal_reward(next_board)
        if score is None:
            score = predict_value(next_board, value_weights)
        scores.append(score)

    best_score = max(scores) if board.turn == chess.WHITE else min(scores)
    best_moves = [
        move for move, score in zip(moves, scores) if np.isclose(score, best_score)
    ]
    return best_moves[int(rng.integers(len(best_moves)))]


def train_one_game(value_weights, epsilon):
    board = chess.Board(START_FEN)
    errors = []
    truncated = False

    for ply in range(MAX_PLIES):
        features = connectome_features(board)
        old_value = float(np.tanh(value_weights @ features))
        old_material = material_score(board)

        board.push(choose_move(board, value_weights, epsilon))

        material_change = (material_score(board) - old_material) / 40.0
        reward = MATERIAL_SHAPING * material_change
        final_reward = terminal_reward(board)
        truncated = ply == MAX_PLIES - 1 and final_reward is None

        if final_reward is not None:
            target = final_reward + reward
        else:
            target = reward + DISCOUNT * predict_value(board, value_weights)

        target = float(np.clip(target, -1.0, 1.0))
        td_error = target - old_value
        value_weights += (
            LEARNING_RATE * td_error * (1.0 - old_value**2) * features
        )
        errors.append(abs(td_error))

        if final_reward is not None:
            break

    result = "draw-limit" if truncated else board.result(claim_draw=True)
    return result, len(board.move_stack), float(np.mean(errors))


def load_value_weights():
    if not save_model_file.is_file():
        return np.zeros(READOUT_NEURONS + 1, dtype=np.float32)

    saved = joblib.load(save_model_file)
    value_weights = np.asarray(saved["value_weights"], dtype=np.float32)
    if value_weights.shape != (READOUT_NEURONS + 1,):
        raise ValueError("The saved readout size does not match READOUT_NEURONS.")

    print(f"Continuing from {save_model_file.name}")
    return value_weights


def opening_test(value_weights):
    board = chess.Board(START_FEN)
    moves = []

    print("\nFull starting position:")
    print(board)
    print(f"FEN: {board.fen()}")

    for _ in range(OPENING_TEST_PLIES):
        if board.is_game_over(claim_draw=True):
            break
        move = choose_move(board, value_weights, epsilon=0.0)
        moves.append(board.san(move))
        board.push(move)

    print("\nFirst self-play moves:")
    for index in range(0, len(moves), 2):
        white_move = moves[index]
        black_move = moves[index + 1] if index + 1 < len(moves) else ""
        print(f"{index // 2 + 1:>2}. {white_move:<8} {black_move}")

    print("\nPosition after those moves:")
    print(board)
    print(f"FEN: {board.fen()}")
    print(f"Learned value: {predict_value(board, value_weights):+.3f}")



value_weights = load_value_weights()

print(f"\ntraining with {SELF_PLAY_GAMES} self-play games")
for game in range(SELF_PLAY_GAMES):
    progress = game / max(SELF_PLAY_GAMES - 1, 1)
    epsilon = EPSILON_START + progress * (EPSILON_END - EPSILON_START)
    result, plies, mean_error = train_one_game(value_weights, epsilon)
    print(
        f"Game {game + 1:>2}/{SELF_PLAY_GAMES}: {result:<10} "
        f"plies={plies:<3} epsilon={epsilon:.2f} TD={mean_error:.4f}"
    )

joblib.dump(
    {
        "value_weights": value_weights,
        "connectome": matrix,
        "selected_neurons": selected,
        "readout_indices": readout_indices,
    },
    save_model_file,
    compress=3,
)
print(f"\n\n\nsaved {save_model_file.name}")
opening_test(value_weights)



