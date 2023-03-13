import torch
from torch import nn
import torch.nn.functional as F 
import torchvision 
from torchvision import transforms 
from torch.utils.data import DataLoader 
from torchsummary import summary
from tqdm import tqdm

# 载入数据集
train_dataset = torchvision.datasets.MNIST(
    root = "dataset/",
    train = True,
    transform = transforms.ToTensor(),
    download = True
)
test_dataset = torchvision.datasets.MNIST(
    root = "dataset/",
    train = False,
    transform = transforms.ToTensor(),
    download = True
)
# 生成dataset
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

# 教师模型
class TeacherModel(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(TeacherModel, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(784,1200)
        self.fc2 = nn.Linear(1200,1200)
        self.fc3 = nn.Linear(1200,num_classes)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.dropout(x)
        x = self.relu(x)

        x = self.fc3(x)
        return x

class StudentModel(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(StudentModel, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(784,20)
        # self.fc2 = nn.Linear(20,20)
        self.fc3 = nn.Linear(20,num_classes)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = self.relu(x)

        # x = self.fc2(x)
        # x = self.relu(x)

        x = self.fc3(x)
        return x




# def main():
#     model = TeacherModel()
#     model = model.to(device)
#     summary(model, (1,28,28))

# if __name__ == '__main__':
#     main()
