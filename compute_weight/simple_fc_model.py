from torch import nn


class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_layers_1=64, hidden_layers_2=128, hidden_layer_3=256, hidden_layer_4=512, hidden_layer_5=1024):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_layers_1)
        self.bn1 = nn.BatchNorm1d(hidden_layers_1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_layers_1, hidden_layers_2)
        self.bn2 = nn.BatchNorm1d(hidden_layers_2)
        self.fc3 = nn.Linear(hidden_layers_2, hidden_layer_3)
        self.bn3 = nn.BatchNorm1d(hidden_layer_3)
        self.fc4 = nn.Linear(hidden_layer_3, hidden_layer_4)
        self.bn4 = nn.BatchNorm1d(hidden_layer_4)
        self.fc5 = nn.Linear(hidden_layer_4, 1)
        # self.bn5 = nn.BatchNorm1d(hidden_layer_5)
        # self.fc6 = nn.Linear(hidden_layer_5, 1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.fc4(out)
        out = self.bn4(out)
        out = self.relu(out)
        out = self.fc5(out)
        # out = self.bn5(out)
        # out = self.relu(out)
        # out = self.fc6(out)
        return out



