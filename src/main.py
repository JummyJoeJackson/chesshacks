# Necessary imports
from .utils import chess_manager, GameContext, EvalNet, encode_board, pgn_to_board
import torch
from huggingface_hub import hf_hub_download
import pathlib


# Write code here that runs once
# Can do things like load models from huggingface, make connections to subprocesses, etcwenis
MODEL_NAME = "JummyJoeJackson/chess-bot-model"
CACHE_DIR = pathlib.Path("./.model_cache")


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
def make_best_move(board, model, depth):
    best_move = None
    best_score = float('-inf')
    for move in board.legal_moves:
        board.push(move)
        score = minimax(board, depth - 1, False, model)
        board.pop()
        if score > best_score:
            best_score = score
            best_move = move
    return best_move


# Load model weights from Hugging Face Hub
def load_model_from_hf():
    # Download and cache model weights from Hugging Face Hub
    CACHE_DIR.mkdir(exist_ok=True)
    if not (CACHE_DIR / MODEL_NAME.split('/')[-1]).exists():
        print(f"Downloading model {MODEL_NAME} from Hugging Face Hub...")
    model = EvalNet()

    # Load weights with huggingface_hub cache if available
    weights_path = hf_hub_download(repo_id=MODEL_NAME, filename="pytorch_model.bin", cache_dir=CACHE_DIR)
    state_dict = torch.load(weights_path)
    model.load_state_dict(state_dict)
    print(f"Model loaded from {weights_path}")
    model.eval()
    return model


# Entrypoint function called to get the next move
@chess_manager.entrypoint
def get_move(pgn: str) -> str:
    board = pgn_to_board(pgn)

    # Get or create cached model
    ctx = chess_manager.context
    model = ctx.state.get('model')
    if model is None:
        try:
            model = load_model_from_hf()
        except Exception as e:
            print("Failed to load model from HF Hub, initializing fresh model", e)
            model = EvalNet()
        ctx.state['model'] = model

    best_move = make_best_move(board, model, depth=3)
    return best_move.uci() if best_move else None


# Reset function called at the start of each new game
@chess_manager.reset
def reset_func(ctx: GameContext):
    try:
        model = load_model_from_hf()
    except Exception as e:
        print("Failed to load model from HF Hub in reset, initializing fresh model", e)
        model = EvalNet()
    ctx.state['model'] = model
