import torch
import torchvision
from torchvision import transforms, datasets

# 载入数据集
import os
import json
data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    "test": transforms.Compose([transforms.RandomResizedCrop(256),
                                 transforms.RandomHorizontalFlip(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
}
data_root = os.path.abspath("/home/gw00243982/gj/a02_code/00_data")
image_path = os.path.join(data_root, "flower_data")
train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
train_num = len(train_dataset)

# {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
flower_list = train_dataset.class_to_idx
cla_dict = dict((val, key) for key, val in flower_list.items())
# write dict into json file
json_str = json.dumps(cla_dict, indent=4)
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)

batch_size = 16
nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
print('Using {} dataloader workers every process'.format(nw))

train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=batch_size, shuffle=True,
                                            num_workers=nw)

test_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                        transform=data_transform["test"])
test_num = len(test_dataset)
test_loader = torch.utils.data.DataLoader(test_dataset,
                                                batch_size=batch_size, shuffle=False,
                                                num_workers=nw)

print("using {} images for training, {} images for test.".format(train_num,
                                                                        test_num))