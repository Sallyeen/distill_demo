```
.
├── class_indices.json # 标签
├── data.py # 数据预处理
├── distill.py # 蒸馏的功能函数
├── model.py # 模型定义
├── model_res.py # resnet模型定义
├── model_v2.py # mobilenet模型定义
├── plt.py # 绘制不同蒸馏温度图
├── pred.py # 模型预测功能函数
├── prune1.py # 剪枝测试函数
├── pth
│   ├── s101.pth
│   ├── s1.pth
│   └── t1.pth
├── s_train.py # 学生网络训练
└── t_train.py # 教师网络训练
```
1. 训练一个teacher模型，得到teacher的pth文件
2. 训练一个student模型，得到student的pth文件
3. 借用teacher对student进行蒸馏	
  a. 用teacher网络进行测试
  b. student进行训练【student输出与标签计算hardloss；student输出与teacher输出计算softloss】