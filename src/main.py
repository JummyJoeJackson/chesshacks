from .utils import *


# Entrypoint function called to get the next move
@chess_manager.entrypoint
def get_move(ctx: GameContext) -> str:
    board = ctx.board
    context = chess_manager._ctx
    model = ctx.state.get('model')

    if model is None:
        try:
            model = load_model(ChessBotConfig())
        except Exception as e:
            print("Failed to load model, initializing fresh model", e)
            model = EvalNet(ChessBotConfig())
        context.state['model'] = model
    
    try:
        best_move = make_best_move(board, model, depth=2)
    except Exception as e:
        print("Error during move calculation:", e)
        best_move = next(iter(board.legal_moves), None)

    if best_move not in board.legal_moves:
        print(f"invalid move generated: {best_move} falling back to random legal move")
        best_move = next(iter(board.legal_moves), None)
        if best_move is None:
            raise ValueError("No legal moves available")
        
    return best_move


# Reset function called at the start of each new game
@chess_manager.reset
def reset_func(ctx: GameContext):
    try:
        model = load_model(ChessBotConfig())
        print("Loaded model successfully in reset")
    except Exception as e:
        print("Failed to load model from in reset, initializing fresh model:", e)
        model = EvalNet(ChessBotConfig())
        torch.save(model.state_dict(), "models/model_weights.pt")
        print("Saved fresh fallback model weights as model_weights.pt")

    ctx.state['model'] = model
