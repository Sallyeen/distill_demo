from multiprocessing import reduction
import torch
from torch import nn
import torch.nn.functional as F 
from torchsummary import summary
from tqdm import tqdm
from model_res import resnet34
from model_v2 import MobileNetV2
from data import train_loader, test_loader
# from model import TeacherModel, StudentModel, train_loader, test_loader

# 随机数种子，便于复现
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True #保证每次运行结果一样
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device: ", device)

# 使用cudnn加速卷积
torch.backends.cudnn.benchmark = True

t_model = resnet34(num_classes=5)
t_model = t_model.to(device)
# summary(model, (1,28,28))
t_model_data = torch.load('pth/t1.pth', map_location=device)
t_model.load_state_dict(t_model_data, strict=False)

t_model.eval()

s_model = MobileNetV2(num_classes=5)
s_model = s_model.to(device)
s_model_data = torch.load('pth/s1.pth', map_location=device)
s_model.load_state_dict(s_model_data, strict=False)
# s_model.train()

temp = 19
alpha = 1/(temp * temp + 1)
# alpha = 1
hard_loss = nn.CrossEntropyLoss()
soft_loss = nn.KLDivLoss(reduction="batchmean")
optimizer = torch.optim.Adam(s_model.parameters(), lr=1e-4)

epochs = 3
best_acc = 0.5522
for epoch in range(epochs):
    s_model.train()

    num_correct = 0
    num_samples = 0
    with torch.no_grad():
        for x,y in test_loader: # x是输入，y是标签
            x = x.to(device)
            y = y.to(device)

            preds = t_model(x)
            predictions = preds.max(1).indices
            num_correct += (predictions == y).sum() # 统计预测对了的个数
            num_samples += predictions.size(0) # 统计预测总数
        acc = (num_correct/num_samples).item() # teacher预测准确率
        print("teacher Accuracy:{:.4f}".format(acc))

    for data, targets in tqdm(train_loader):
        data = data.to(device)
        targets = targets.to(device)

        student_preds = s_model(data)
        label_loss = hard_loss(student_preds, targets)

        with torch.no_grad():
            teacher_preds = t_model(data)
            distill_loss = soft_loss(
            F.softmax(student_preds / temp, dim=1),
            F.softmax(teacher_preds / temp, dim=1)
        )

        loss = alpha * label_loss + (1-alpha)*distill_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    s_model.eval()
    num_correct = 0
    num_samples = 0
     
    with torch.no_grad():
        for x,y in test_loader:
            x = x.to(device)
            y = y.to(device)

            preds = s_model(x)
            predictions = preds.max(1).indices
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        acc = (num_correct/num_samples).item()

    if acc > best_acc:
        best_acc = acc
        torch.save(s_model.state_dict(), 'pth/s101.pth')
        print("Sucsessful!")

    s_model.train()
    print("Epoch:{}\t Accuracy:{:.4f}".format(epoch+1, acc))

