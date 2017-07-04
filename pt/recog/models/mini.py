import torch.nn as nn
import torch.nn.functional as F

MINI = 'mini'


class Mini(nn.Module):
    def __init__(self, input_shape):
        super(Mini, self).__init__()
        self.nchans, self.nrows, self.ncols = input_shape
        self.conv1 = nn.Conv2d(self.nchans, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()

        self.final_nrows = (((self.nrows - 4) / 2) - 4) / 2
        self.final_ncols = (((self.nrows - 4) / 2) - 4) / 2
        self.final_conv_sz = int(self.final_nrows * self.final_ncols * 20)

        self.fc1 = nn.Linear(self.final_conv_sz, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, self.final_conv_sz)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)
