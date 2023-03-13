import torch
from torch import nn
import torch.nn.functional as F 
from torchsummary import summary
from tqdm import tqdm
from model_res import resnet34
from data import train_loader, test_loader
from torchstat import stat
# from model import StudentModel, train_loader, test_loader

# 随机数种子，便于复现
torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 使用cudnn加速卷积
torch.backends.cudnn.benchmark = True
# num_classes=5
model2 = resnet34(num_classes=5)
# model2 = model2.to(device)
# summary(model2, (3, 224, 224))
stat(model2, (3, 224, 224))
model2 = model2.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model2.parameters(), lr=1e-4)

epochs = 3
best_acc =0.6511
for epoch in range(epochs):
    model2.train()

    for data, targets in tqdm(train_loader):
        data = data.to(device)
        targets = targets.to(device)

        preds = model2(data)
        # print("epoch", epoch, "----", targets)
        # print("epoch", epoch, "----", preds)
        loss = criterion(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model2.eval()
    num_correct = 0
    num_samples = 0
     
    with torch.no_grad():
        for x,y in test_loader:
            x = x.to(device)
            y = y.to(device)

            preds = model2(x)
            predictions = preds.max(1).indices
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        acc = (num_correct/num_samples).item()

    if acc > best_acc:
        best_acc = acc
        torch.save(model2.state_dict(), 'pth/t1.pth')
        print("Sucsessful!")

    model2.train()
    print("Epoch:{}\t Accuracy:{:.4f}".format(epoch+1, acc))
