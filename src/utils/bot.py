import json
import torch
from torch import nn, optim
from torch.utils.data import IterableDataset, DataLoader
from transformers import PretrainedConfig, PreTrainedModel
from .utils import fen_to_board, encode_board


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


# Training loop for the neural network
def train_eval_net(eval_net, dataset, epochs=5, batch_size=32, lr=0.001):
    dataloader = DataLoader(dataset, batch_size=batch_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(eval_net.parameters(), lr=lr)

    for epoch in range(epochs):
        loss = None
        for board_tensors, evaluations in dataloader:
            optimizer.zero_grad()
            outputs = eval_net(board_tensors)
            loss = criterion(outputs, evaluations)
            loss.backward()
            optimizer.step()
        if loss is not None:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
        else:
            print(f"Epoch {epoch+1}/{epochs}, No data processed.")


# Example usage
if __name__ == "__main__":
    # Initialize the evaluation network
    config = ChessBotConfig()
    eval_net = EvalNet(config)

    # Prepare dataset
    file_list = ['Training/lichess_db_eval_part1_simplified.jsonl', 'Training/lichess_db_eval_part1_simplified.jsonl']  # Actual file paths
    dataset = EvalDataset(file_list, encode_board)

    # saves the model weights after training
    torch.save(eval_net.state_dict(), "model_weights.pt")

    # Train the evaluation network
    train_eval_net(eval_net, dataset, epochs=10, batch_size=64, lr=0.0005)