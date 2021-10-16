import torch.nn as nn


class Net(nn.Module):
    def __init__(self, hidden_size, embedding_dim, vocab_size):
        super(Net, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = nn.LSTM(
            input_size=embedding_dim, hidden_size=hidden_size, num_layers=1
        )
        self.predictor = nn.Linear(hidden_size, 1)

    def forward(self, seq):
        _, (hidden, _) = self.encoder(self.embedding(seq))
        preds = self.predictor(hidden.squeeze(0))
        return preds
