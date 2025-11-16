import chess
import torch
import os

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
model_path = os.path.join(root_dir, "models", "model_weights.pt")


# Convert FEN string to a chess.Board object
def fen_to_board(fen: str) -> chess.Board:
    board = chess.Board(fen)
    return board


# Load model from Local Storage
def load_model(config):
    from .bot import EvalNet
    model = EvalNet(config)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print("Loaded saved model weights from", model_path)
    else:
        print("No saved model weights found at", model_path)
    model.eval()
    return model


# Encodes chess board to tensor (64 squares * 6 piece types)
def encode_board(board: chess.Board) -> torch.Tensor:
    piece_map = board.piece_map()
    vector = torch.zeros(64 * 6)
    piece_to_idx = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5,
    }
    for square, piece in piece_map.items():
        offset = 6 * square
        idx = piece_to_idx[piece.piece_type]
        vector[offset + idx] = 1 if piece.color == chess.WHITE else -1
    return vector


# Minimax search using trained model
def minimax(board, depth, maximizing, model, alpha=float('-inf'), beta=float('inf')):
    # Base case: evaluate board using neural network
    if depth == 0 or board.is_game_over():
        x = encode_board(board)
        with torch.no_grad():
            return model(x).item()
    moves = list(board.legal_moves)
    # Recursive case: explore moves
    if maximizing:
        max_eval = float('-inf')
        for move in moves:
            board.push(move)
            eval_score = minimax(board, depth - 1, False, model, alpha, beta)
            board.pop()
            if eval_score > max_eval:
                max_eval = eval_score
            if max_eval > alpha:
                alpha = max_eval
            if beta <= alpha:
                break
        return max_eval
    # Minimizing player
    else:
        min_eval = float('inf')
        for move in moves:
            board.push(move)
            eval_score = minimax(board, depth - 1, True, model, alpha, beta)
            board.pop()
            if eval_score < min_eval:
                min_eval = eval_score
            if min_eval < beta:
                beta = min_eval
            if beta <= alpha:
                break
        return min_eval


# Select the best move using minimax and the neural network
def make_best_move(board, model, depth):
    best_move = None
    best_score = float('-inf')

    for move in board.legal_moves:
        board.push(move)
        try:
            score = minimax(board, depth - 1, False, model)
        except Exception as e:
            print("Error during minimax evaluation:", e)
            score = float('-inf')
        board.pop()
        if score > best_score:
            best_score = score
            best_move = move
    return best_move
