from .utils import *


MODEL_LOCAL_PATH = "./model_weights.pt"
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

# Test function to train the model (not called during gameplay)
if __name__ == "__main__":
    board = chess.Board()
    config = ChessBotConfig()
    model = EvalNet(config)

    board.push_san("e4")
    board.push_san("e5")
    board.push_san("Nf3")
    board.push_san("Nc6")

    move = get_move(GameContext(board, timeLeft=300, logProbabilities=None ))

    print("Model output:", move)
