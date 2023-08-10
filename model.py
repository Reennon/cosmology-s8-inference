from torch import nn


class one_hidden_layer(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, dropout_rate):
        super(one_hidden_layer, self).__init__()

        # define the fully connected layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

        # define the other layers
        self.dropout = nn.Dropout(p=dropout_rate)
        self.ReLU = nn.ReLU()
        self.LeakyReLU = nn.LeakyReLU(0.2)

    # forward pass
    def forward(self, x):
        out = self.fc1(x)
        out = self.LeakyReLU(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out
