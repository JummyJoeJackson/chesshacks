import json
import torch
from torch import nn, optim
from torch.utils.data import IterableDataset, DataLoader
from transformers import PretrainedConfig, PreTrainedModel
from utilities import *


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
    #from .utilities import fen_to_board

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


# Training loop for the neural network
def train_model(model, dataset, epochs=5, batch_size=64, lr=0.0005):
    dataloader = DataLoader(dataset, batch_size=batch_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        loss = None
        for board_tensors, evaluations in dataloader:
            optimizer.zero_grad()
            outputs = model(board_tensors)
            loss = criterion(outputs, evaluations)
            loss.backward()
            optimizer.step()
        if loss is not None:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
        else:
            print(f"Epoch {epoch+1}/{epochs}, No data processed.")


# Example usage
if __name__ == "__main__":
    #from .utilities import load_model, encode_board

    model = load_model(ChessBotConfig())

    file_list = ['Training/evals/lichess_db_eval_part1_simplified.jsonl', 'Training/evals/lichess_db_eval_part1_simplified.jsonl']
    dataset = EvalDataset(file_list, encode_board)

    train_model(model, dataset, epochs=1, lr=0.0005)

    torch.save(model.state_dict(), "models/model_weights.pt")
    print("Model training complete and weights saved to models/model_weights.pt")
