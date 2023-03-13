from model_res import resnet34
import torch
import torch.nn.utils.prune as prune

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnet34(num_classes=5)
# for i, module in enumerate(model.state_dict()):
#     print('----')
#     print(i)
#     print('--------')
#     print(module)
#     prune.ln_structured(module, name="weight", amount=0.33, n=2, dim=0)
#     prune.remove(module, 'weight')
#     print('-------1-------')
#     print(list(module.named_parameters()))  # 剪枝层orig
#     print('-------2-------')
#     print(list(module.named_buffers()))  # 剪枝层mask
#     print('-------3-------')
#     print(model.state_dict().keys())  # 剪枝层名称
prune.ln_structured(model.layer1, name="weight", amount=0.33, n=2, dim=0)
prune.remove(module, 'weight')