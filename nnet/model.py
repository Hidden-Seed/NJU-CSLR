import torch
import torch.nn as nn


class model(nn.Module):
    def __init__(self, OUTPUT_SIZE, DROP_RATE, INPUT_SIZE=[84, 256, 512], HIDDEN_SIZE=[64, 128, 64]):
        super(model, self).__init__()

        # Dropout layer
        self.dropout = nn.Dropout(DROP_RATE)

        # 第 1 层 BiLSTM
        self.rnn1 = nn.LSTM(
            input_size=INPUT_SIZE[0],
            hidden_size=HIDDEN_SIZE[0],
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.fc1 = nn.Linear(2 * HIDDEN_SIZE[0], INPUT_SIZE[1])

        # 第 2 层 BiLSTM
        self.rnn2 = nn.LSTM(
            input_size=INPUT_SIZE[1],
            hidden_size=HIDDEN_SIZE[1],
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.fc2 = nn.Linear(2 * HIDDEN_SIZE[1], INPUT_SIZE[2])

        # 第 3 层 BiLSTM
        self.rnn3 = nn.LSTM(
            input_size=INPUT_SIZE[2],
            hidden_size=HIDDEN_SIZE[2],
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.fc3 = nn.Linear(2 * HIDDEN_SIZE[2], OUTPUT_SIZE)

    def forward(self, frames_batch):
        # 第 1 层
        r_out1, _ = self.rnn1(frames_batch)
        out1 = self.dropout(torch.relu(self.fc1(r_out1)))

        # 第 2 层
        r_out2, _ = self.rnn2(out1)
        out2 = self.dropout(torch.relu(self.fc2(r_out2)))

        # 第 3 层
        r_out3, _ = self.rnn3(out2)
        out3 = self.fc3(r_out3)

        return out3
