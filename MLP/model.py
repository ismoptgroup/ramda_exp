import torch

class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(in_features=784, out_features=512)
        self.fc2 = torch.nn.Linear(in_features=512, out_features=256)
        self.fc3 = torch.nn.Linear(in_features=256, out_features=128)
        self.fc4 = torch.nn.Linear(in_features=128, out_features=64)
        self.fc5 = torch.nn.Linear(in_features=64, out_features=32)
        self.fc6 = torch.nn.Linear(in_features=32, out_features=16)
        self.fc7 = torch.nn.Linear(in_features=16, out_features=10)
        self.relu = torch.nn.ReLU(inplace=True)
        self.log_softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.relu(self.fc6(x))
        x = self.fc7(x)
        return self.log_softmax(x)