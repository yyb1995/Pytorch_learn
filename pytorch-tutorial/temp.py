import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np

# specify the device to run on
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper parameters
sequence_length = 20
input_size = 28
hidden_size = 128
num_layers = 2
output_length = 1
batch_size = 100
num_epochs = 2
learning_rate = 0.01

# data pre-processing
x_input = torch.sin(torch.Tensor([i / 2000 * 6 * np.pi for i in range(2000)], device=device))
y_output = torch.sin(torch.Tensor([i / 2000 * 6 * np.pi for i in range(1, 2001)], device=device))


# define RNN network. Module should rewrite two methods. one is __init__,
# the other one is forward
class RNN(nn.Module):
    def __init__(self, input_len, hidden_size, num_layers, output_length):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_len, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, output_length)

    def forward(self, x):
        # lstm_output: seq_length, batch_size, hidden_size
        out, _ = self.lstm(x)
        output = self.fc(out[-1, :, :])
        return out


# to decide which device to run on, use .to(device) to specify
model = RNN(input_size, hidden_size, num_layers, output_length).to(device)

# Loss and optimizer
loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

# train model
for epoch in range(num_epochs):
    y_predict = model(x_input)
    loss = loss_func(y_output, y_predict)
    optimizer.zero_grad()
    optimizer.step()