from .utils import *
import json
import torch
import chess
from torch import nn, optim
from torch.utils.data import IterableDataset, DataLoader
from transformers import PretrainedConfig, PreTrainedModel
from .utils import fen_to_board, encode_board


MODEL_GDRIVE_ID = ""
MODEL_LOCAL_PATH = "./Training/model_weights.pt"
config = ChessBotConfig()


# Entrypoint function called to get the next move
@chess_manager.entrypoint
def get_move(ctx: GameContext) -> str:
    board = ctx.board
    
    # Get or create cached model
    ctx = chess_manager._ctx
    model = ctx.state.get('model')
    config = ChessBotConfig()
    if model is None:
        try:
            model = load_model(config)
        except Exception as e:
            print("Failed to load model, initializing fresh model", e)
            model = EvalNet(config)
        ctx.state['model'] = model
    
    # Determine best move using minimax with the model
    try:
        best_move = make_best_move(board, model, depth=3)
    except Exception as e:
        print("Error during move calculation:", e)
        # Fallback: make a random legal move
        best_move = list(board.legal_moves)[0]

    # Ensure best_move is not None
    if not best_move or best_move not in board.legal_moves:
        print(f"invalid move generated: {best_move} falling back to random move")
        best_move = next(iter(board.legal_moves), None)
        if best_move is None:
            raise ValueError("No legal moves available")
    return best_move


# Reset function called at the start of each new game
@chess_manager.reset
def reset_func(ctx: GameContext):
    try:
        model = load_model(config)
    except Exception as e:
        print("Failed to load model from in reset, initializing fresh model", e)
        model = EvalNet(config)
        torch.save(model.state_dict(), "model_weights.pt")
        print("Saved fresh fallback model weights as model_weights.pt")
    ctx.state['model'] = model


# Define a config class (required for compatibility)
class ChessBotConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


# Define the evaluation neural network
class EvalNet(PreTrainedModel):
    config_class = ChessBotConfig

    def __init__(self, config):
        super().__init__(config)
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(64 * 6, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.layers(x)


# Dataset for loading evaluation data
class EvalDataset(IterableDataset):
    def __init__(self, file_list, encode_board):
        self.file_list = file_list
        self.encode_board = encode_board

    def line_iterator(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                yield line

    def __iter__(self):
        for file_path in self.file_list:
            for line in self.line_iterator(file_path):
                try:
                    data = json.loads(line)
                    board = fen_to_board(data['fen'])
                    board_tensor = self.encode_board(board)
                    evaluation = torch.tensor([data['score_cp']], dtype=torch.float32)
                    yield board_tensor, evaluation
                except (json.JSONDecodeError, KeyError):
                    continue


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
