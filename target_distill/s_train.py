import torch
from torch import nn
import torch.nn.functional as F  
from torchsummary import summary
from tqdm import tqdm
# from model import TeacherModel, train_loader, test_loader
from model_v2 import MobileNetV2
from data import train_loader, test_loader

# 随机数种子，便于复现
torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 使用cudnn加速卷积
torch.backends.cudnn.benchmark = True

model = MobileNetV2(num_classes=5)
model = model.to(device)
summary(model, (3,224,224))

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

epochs = 3
best_acc = 0.5220
for epoch in range(epochs):
    model.train()

    for data, targets in tqdm(train_loader):
        data = data.to(device)
        targets = targets.to(device)

        preds = model(data)
        loss = criterion(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    num_correct = 0
    num_samples = 0
     
    with torch.no_grad():
        for x,y in test_loader:
            x = x.to(device)
            y = y.to(device)

            preds = model(x)
            predictions = preds.max(1).indices
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        acc = (num_correct/num_samples).item()

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), 'pth/s1.pth')
        print("Sucsessful!")

    model.train()
    print("Epoch:{}\t Accuracy:{:.4f}".format(epoch+1, acc))