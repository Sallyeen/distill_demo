from multiprocessing import reduction
import argparse
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

# def parse_args():
#     parser = argparse.ArgumentParser(description='model prediction')
#     parser.add_argument('--model', help='model to predict', type=str, default='teacher')

def pred():
    # args = parse_args()
    model = model_name(num_classes=5)
    model = model.to(device)
    # summary(model, (1,28,28))
    model_data = torch.load(weight_path, map_location=device)
    model.load_state_dict(model_data, strict=False)

    model.eval()
    num_correct = 0
    num_samples = 0
    with torch.no_grad():
        for x,y in test_loader: # x是输入，y是标签
            x = x.to(device)
            y = y.to(device)

            preds = model(x)
            predictions = preds.max(1).indices
            num_correct += (predictions == y).sum() # 统计预测对了的个数
            num_samples += predictions.size(0) # 统计预测总数
        acc = (num_correct/num_samples).item() # teacher预测准确率
        print("model {} Accuracy:{:.4f}".format(model_name, acc))


if __name__ == '__main__':
    model_name = resnet34
    weight_path = 'pth/t1.pth'
    pred()
    print('--------------------------------')
    model_name = MobileNetV2
    weight_path = 'pth/s1.pth'
    pred()