import torch.nn as nn
import torch.nn.functional as F

class ImageCLFNet(nn.Module):
    def __init__(self):
        super(ImageCLFNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 30, kernel_size=3)
        self.conv2 = nn.Conv2d(30, 60, kernel_size=3)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(60 * 17 * 12, 50)
        self.fc2 = nn.Linear(50, 7)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        
        # batch lost here
        # print(x.shape)
        x = x.view(-1, 60 * 17 * 12)
        
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)