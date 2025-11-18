import modal
import sys

app = modal.App("example-get-started")


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers", "numpy", "chess")
    .add_local_file("utilities.py", "/app/utilities.py")
    .add_local_file("bot.py", "/app/bot.py")
    .add_local_file("utilities.py", "/app/utilities.py")
    # add all files located in /Training/evals
    .add_local_file("evals/lichess_db_eval_part1_simplified.jsonl", "/app/lichess_db_eval_part2_simplified.jsonl")
    .add_local_file("models/model_weights.pt", "/app/models/model_weights.pt")
)


@app.function(image=image, gpu="A100-40GB", timeout=3600)
def train_chess_model():
    import torch
    import json
    from torch import nn, optim
    from torch.utils.data import IterableDataset, DataLoader
    import torch.optim.lr_scheduler as lr_scheduler
    from transformers import PretrainedConfig, PreTrainedModel
 
    sys.path.append("/app")
    from utilities import load_model, encode_board, fen_to_board
    
    class ChessBotConfig(PretrainedConfig):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

    class EvalNet(PreTrainedModel):
        config_class = ChessBotConfig

        def __init__(self, config):
            super().__init__(config)
            self.layers = nn.Sequential(
                nn.Linear(64 * 6, 128),
                nn.ReLU(),
                nn.Linear(128, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )

        def forward(self, x):
            return self.layers(x)

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

    def train_model(model, dataset, epochs=1, batch_size=32, lr=0.001):
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, pin_memory=True)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
        model.train()

        for epoch in range(epochs):
            running_loss = 0.0
            count = 0
            for board_tensors, evaluations in dataloader:
                board_tensors = board_tensors.to(device) # Move data to the GPU
                evaluations = evaluations.to(device) # Move data to the GPU

                optimizer.zero_grad()
                outputs = model(board_tensors)
                loss = criterion(outputs, evaluations)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                count += 1
                print(f"Batch: {count}/12896")
            scheduler.step()

            avg_loss = running_loss / count if count > 0 else float('inf')
            print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")

        torch.save(model.state_dict(), "model_weights.pt")
        print("Training complete and weights saved to model_weights.pt")
        return "Finished"

    # List your data files that are in /Training/evals
    file_list = ['/app/lichess_db_eval_part1_simplified.jsonl', "/app/lichess_db_eval_part2_simplified.jsonl"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(ChessBotConfig())
    model.to(device)
    dataset = EvalDataset(file_list, encode_board)
    return train_model(model, dataset, epochs=5, batch_size=256, lr=0.001)


@app.local_entrypoint()
def main():
    result = train_chess_model.remote()
    print(result)
