import torch
from torch.utils.data import Dataset, DataLoader
import random
class_list = ['dog', 'cat','bear','sheep','bird','rat','horse','elephant','cow','monkey']
from PIL import Image
import numpy as np
import os
import torchvision.transforms as transforms
class NICO_dataset(torch.utils.data.Dataset):
    def __init__(self, all_data, all_label,image_paths): # 通过label值确定content  data,
        super(NICO_dataset, self).__init__()
        self.all_data = all_data
        # self.data=data  # frame data
        self.all_label = all_label  # id值固定  dog为0，cat为1
        self.image_paths=image_paths
        self.transform = transforms.ToTensor()  # 转换为张量

    def __getitem__(self, item):  # 每一个item值对应一张图片,从0开始
        img = self.all_data[item]
        img = self.transform(img)
        # img_frame=self.data[item]

        label = self.all_label   # self.all_label[item]
        content=class_list[label]  # 0为dog
        path = self.image_paths[item]  # 获取图像路径

        return img, label,content,path

    def __len__(self):
        return len(self.all_data)

class NICO_dataset_2(torch.utils.data.Dataset):
    def __init__(self, all_data, all_label):
        super(NICO_dataset_2, self).__init__()
        self.all_data = all_data

        self.all_label = all_label


    def __getitem__(self, item):
        img = self.all_data[item]


        label = self.all_label[item]


        return img, label

    def __len__(self):
        return len(self.all_data)
    
class NICO_dataset_3(torch.utils.data.Dataset):
    def __init__(self, all_data, all_label):
        super(NICO_dataset_3, self).__init__()
        self.all_data = all_data
        self.all_label = all_label
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.52418953, 0.5233741, 0.44896784],
                                 std=[0.21851876, 0.2175944, 0.22552039])
        ])

    def __getitem__(self, item):
        img = self.all_data[item]
        img=self.transform(img)
        label = self.all_label[item]

        return img,label

    def __len__(self):
        return len(self.all_data)

def get_NICO_dataloader_train(id):  # 通过id值确定label
    seed = 1000
    torch.manual_seed(seed)
    random.seed(seed)
    # original data
    # loaded_data = torch.load('NICO2/nico_client{}.pt'.format(id+1))
    # loaded_label=torch.load('NICO2/nico_client{}_label.pt'.format(id+1))
    # frame data
    # frame_data = torch.load('NICO_frame/nico_client{}.pt'.format(id + 1))
    # frame_label = torch.load('NICO_frame/nico_client{}_label.pt'.format(id + 1))
    base_dir = '../Datasets/NICO'  # 把这里改成每一个动物的文件夹名  

    # Find the folder that starts with the given id
    folder_name = None
    for folder in os.listdir(base_dir):
        if folder.startswith(str(id)):
            folder_name = folder  # 对应id值的文件夹，比如说狗的文件夹(id=0)
            break

    if folder_name is None:
        raise FileNotFoundError(f"No folder starting with {id} found in {base_dir}")

    # Load data from the found folder  狗文件夹下有多个子文件夹
    loaded_data_dir = os.path.join(base_dir, folder_name)  #0dog
    loaded_label = id   # folder_name[len(str(id)):]  # label值   0
    
    loaded_data = []
    contents = []
    img_set = []  # 用于存储加载的图片
    image_paths = []  # 用于存储图片路径

    # Traverse the inner folders in loaded_data_dir
    for subfolder in os.listdir(loaded_data_dir):  # 对子文件夹进行遍历 （0on grass）
        subfolder_path = os.path.join(loaded_data_dir, subfolder)  # 每一个子文件夹路径
        if os.path.isdir(subfolder_path):
            """
            # Split the subfolder name into number and description
            parts = subfolder.split('_', 1)
            if len(parts) == 2:
                number = parts[0]  # The first part is the number
                description = parts[1]  # The second part is the description
                loaded_data.append(subfolder_path)  # 子文件夹路径
                contents.append((number, description))  # 编号+content
                
                # Get image paths from the subfolder
                """
            for filename in os.listdir(subfolder_path):  # 各个图片
                    if filename.endswith(('.png', '.jpg', '.jpeg')):  # Add more formats if needed
                        img_path = os.path.join(subfolder_path, filename)
                        image_paths.append(img_path)  # 存储图片路径

                        # 加载图片
                        img = Image.open(img_path).convert('RGB')  # 加载并转换为RGB格式
                        img_set.append(img)  # 存储加载的图片

    # Get image paths from the found folder
    """
    image_paths = [os.path.join(loaded_data_dir, filename)
                   for filename in os.listdir(loaded_data_dir)
                   if filename.endswith(('.png', '.jpg', '.jpeg'))]  # Add more formats if needed
    """
    dataset=NICO_dataset(img_set,loaded_label,image_paths)  #  图片、id、每一个图片的路径  
    train_dl=DataLoader(dataset,batch_size=1,shuffle=True)  # ==64
    return train_dl

def get_NICO_dataloader_test():
    seed = 1000
    torch.manual_seed(seed)
    random.seed(seed)
    #labels
    test_data=torch.load('NICO_test/nico_test.pt')
    test_labels=torch.load('NICO_test/nico_test_label.pt')
    dataset_test=NICO_dataset_2(test_data,test_labels)
    test_dl=DataLoader(dataset_test,batch_size=64,shuffle=False)

    return test_dl