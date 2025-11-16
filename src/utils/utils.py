import chess
import torch
from .bot import *


# Configuration for the chess bot model
MODEL_LOCAL_PATH = "./model_weights.pt"


# Convert FEN string to a chess.Board object
def fen_to_board(fen: str) -> chess.Board:
    board = chess.Board(fen)
    return board


# Load model from Google Drive
def load_model(config):
    model = EvalNet(config)
    state_dict = torch.load(MODEL_LOCAL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
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
def minimax(board, depth, maximizing, model):
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
            eval_score = minimax(board, depth - 1, False, model)
            board.pop()
            if eval_score > max_eval:
                max_eval = eval_score
        return max_eval
    # Minimizing player
    else:
        min_eval = float('inf')
        for move in moves:
            board.push(move)
            eval_score = minimax(board, depth - 1, True, model)
            board.pop()
            if eval_score < min_eval:
                min_eval = eval_score
        return min_eval


# Select the best move using minimax and the neural network
def make_best_move(board, model, depth) -> chess.Move:
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
