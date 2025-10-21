import os
import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from PIL import Image
import ipdb
import cv2
def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass
    
def build_client(n_env,dataroot,n_labels):  # 获取背景图片  n_env=7  dataroot='./dino/NICO_Contextual' n_labels=10
    '''
    return:image_env:[(array,dtype=uint8)......]
           label_env=[(array)......]
    '''
    image_env = []
    label_env = []
    #context_env = []
    env_cont = []
    for l in range(n_labels):
        env_cont.append(range(10))#十类动物标签
    for env_idx in range(n_env):#七个客户端
        image_env.append([])
        label_env.append([])
        #context_env.append([])
    label_names = os.listdir(dataroot) # 0dog 1cat 
    label_names = sorted(label_names)
#     print('build_client',label_names)
    for label_idx in range(n_labels): # 对于每一个类别
        context_names = os.listdir(dataroot +'/'+label_names[label_idx] + '/')
        context_names = sorted(context_names)
        for env_idx in range(n_env):  # 对于每一个类别中的每一个上下文
            context_idx = env_cont[label_idx][env_idx]
            for folder_name in context_names:
                if folder_name[0] == str(context_idx):
                    path = dataroot + '/'+label_names[label_idx] + '/' + context_names[context_idx] + '/'  # 标签+背景
                    for folder_name1 in os.listdir(path): # 最下层的文件夹
                        folder_path = os.path.join(path, folder_name1)
                        for file_name in os.listdir(folder_path):
                             # 检查文件名是否以 "1_" 开头
                                if file_name.startswith('1_'):
                                    file_path = os.path.join(folder_path, file_name) # 图像路径
                                    imag=Image.open(file_path)
                                    # imag=cv2.resize(imag,(224,224))
                                    image_env[env_idx].append(imag)
                                    label_env[env_idx].append(label_idx)
                    """
                    #print(path)
                    image_names = os.listdir(path)
                    image_names = sorted(image_names)
        #             print('build_client',image_names)
                    #for img in image_names:
                        #temp=transform(Image.open(os.path.join(path, img)).convert('RGB'))
                    for img in image_names:  # 对于每一个类别中的每一个上下文中的每一张图像
                        #imag=cv2.imread(os.path.join(path, img))
                        #imag=cv2.resize(imag,(224,224))
                        imag=Image.open(os.path.join(path,img)).resize((224,224))
                        image_env[env_idx].append(imag)
                        label_env[env_idx].append(label_idx)
                    """
    labels = [np.array(client_label) for client_label in label_env]
    
    for i in range(len(image_env)):
        print(len(image_env[i]), len(labels[i]))
    return image_env, labels