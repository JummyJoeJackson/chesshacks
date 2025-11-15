# Necessary imports
import chess
import torch
import json
import os
import shutil
from torch import nn, optim
from torch.utils.data import IterableDataset, DataLoader
from huggingface_hub import HfApi, Repository


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


# Define the neural network architecture
class EvalNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(64 * 6, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
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
                    board_tensor = self.encode_board(data['pgn'])
                    evaluation = torch.tensor([data['evaluation']], dtype=torch.float32)
                    yield board_tensor, evaluation
                except (json.JSONDecodeError, KeyError):
                    continue


# Training loop for the neural network
def train_eval_net(eval_net, dataset, epochs=5, batch_size=32, lr=0.001):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(eval_net.parameters(), lr=lr)

    for epoch in range(epochs):
        for board_tensors, evaluations in dataloader:
            optimizer.zero_grad()
            outputs = eval_net(board_tensors)
            loss = criterion(outputs, evaluations)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")


if __name__ == "__main__":
    # Example usage
    model = EvalNet()
    dataset = EvalDataset(['data/eval_data.jsonl'], encode_board)
    train_eval_net(model, dataset)

    # Save model locally first
    model_path = "eval_net_weights.pt"
    torch.save(model.state_dict(), model_path)
    
    # Push to HF Hub
    repo_name = "JummyJoeJackson/chess-bot-model"
    repo_local_dir = "./chess-bot-model"

    # Clone or create repo locally
    api = HfApi()
    if not os.path.exists(repo_local_dir):
        api.create_repo(repo_name, exist_ok=True)
        repo = Repository(local_dir=repo_local_dir, clone_from=repo_name)
    else:
        repo = Repository(local_dir=repo_local_dir)
    
    # Copy weights to repo folder
    shutil.copy(model_path, repo_local_dir)

    # Commit and push
    repo.git_add()
    repo.git_commit("Update model weights after training")
    repo.git_push()
    
    print(f"Model pushed to Hugging Face Hub at https://huggingface.co/{repo_name}")
