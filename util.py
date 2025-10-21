import os
import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from PIL import Image
import ipdb
from pathlib import Path
from kmeans import *
import random
from collections import defaultdict
import clip
# import cv2
def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass

def kl_divergence(p, q,device):
    kl_loss = torch.nn.KLDivLoss(reduction='batchmean').to(device)
    log_p = F.log_softmax(p, dim=1)
    q = F.softmax(q, dim=1)
    kl_div = kl_loss(log_p, q)
    return kl_div

def compute_accuracy(model, dataloader, args, get_confusion_matrix=False, device="cpu"):
    was_training = False
    if model.training:
        model.eval()
        was_training = True

    true_labels_list, pred_labels_list = np.array([]), np.array([])

    correct, total = 0, 0
    if device == 'cpu':
        criterion = nn.CrossEntropyLoss()
    elif "cuda" in device.type:
        criterion = nn.CrossEntropyLoss().cuda()
    loss_collector = []
    
#     img_features_all = []
#     img_targets_all = []
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(dataloader):
            #print("x:",x)
            if device != 'cpu':
                x, target = x.cuda(), target.to(dtype=torch.int64).cuda()
                model = model.cuda()
            feats,out = model(x)
            #out=model(x)
            loss = criterion(out, target)
            _, pred_label = torch.max(out.data, 1)
            loss_collector.append(loss.item())
            total += x.data.size()[0]
            correct += (pred_label == target.data).sum().item()

            if device == "cpu":
                pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                true_labels_list = np.append(true_labels_list, target.data.numpy())
            else:
                pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())

#             img_features_all.append(feats)
#             img_targets_all.extend(target) 
#         features_all = torch.cat(img_features_all).cpu().numpy()
#         targets_all = torch.tensor(img_targets_all).cpu().numpy()
        
        avg_loss = sum(loss_collector) / len(loss_collector)

    if get_confusion_matrix:
        conf_matrix = confusion_matrix(true_labels_list, pred_labels_list)

    if was_training:
        model.train()

#     mkdirs('./feats/NICO_Vehicle/fedavg/49/')
#     np.save('./feats/NICO_Vehicle/fedavg/49' + '/local_' + str(net_id) + '_imgfeats.npy', features_all)
#     np.save('./feats/NICO_Vehicle/fedavg/49' + '/local_' + str(net_id) + '_imglabels.npy', targets_all)   
    if get_confusion_matrix:
        return correct / float(total), conf_matrix, avg_loss
 
    return correct / float(total), avg_loss

def compute_accuracy_plan3(model, dataloader, get_confusion_matrix=False, device="cpu", multiloader=False):
    was_training = False
    if model.training:
        model.eval()
        was_training = True

    true_labels_list, pred_labels_list = np.array([]), np.array([])

    correct, total = 0, 0
    if device == 'cpu':
        criterion = nn.CrossEntropyLoss()
    elif "cuda" in device.type:
        criterion = nn.CrossEntropyLoss().cuda()
    loss_collector = []
    if multiloader:
        for loader in dataloader:
            with torch.no_grad():
                for batch_idx, (x, target) in enumerate(loader):
                    #print("x:",x)
                    #print("target:",target)
                    if device != 'cpu':
                        x, target = x.cuda(), target.to(dtype=torch.int64).cuda()
                    outputs = model(x)
                    out = outputs[-1]
                    if len(target)==1:
                        loss = criterion(out, target)
                    else:
                        loss = criterion(out, target)
                    _, pred_label = torch.max(out[:,:10].data, 1)
                    loss_collector.append(loss.item())
                    total += x.data.size()[0]
                    correct += (pred_label == target.data).sum().item()

                    if device == "cpu":
                        pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                        true_labels_list = np.append(true_labels_list, target.data.numpy())
                    else:
                        pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                        true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())
        avg_loss = sum(loss_collector) / len(loss_collector)
    else:
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(dataloader):
                #print("x:",x)
                if device != 'cpu':
                    x, target = x.cuda(), target.to(dtype=torch.int64).cuda()
                outputs = model(x)
                out = outputs[-1]
                #out=model(x)
                loss = criterion(out, target)
                _, pred_label = torch.max(out[:,:10].data, 1)
                loss_collector.append(loss.item())
                total += x.data.size()[0]
                correct += (pred_label == target.data).sum().item()

                if device == "cpu":
                    pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                    true_labels_list = np.append(true_labels_list, target.data.numpy())
                else:
                    pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                    true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())
            avg_loss = sum(loss_collector) / len(loss_collector)

    if get_confusion_matrix:
        conf_matrix = confusion_matrix(true_labels_list, pred_labels_list)

    if was_training:
        model.train()

    if get_confusion_matrix:
        return correct / float(total), conf_matrix, avg_loss

    return correct / float(total), avg_loss

def gen_proto_local_WWW2(net, dataloader, n_class=10):
    feats = []
    labels = []
    net.eval()
    net.cpu()
    with torch.no_grad():
        for batch_idx, (x, obj, back, target) in enumerate(dataloader):
            feat, _ = net(x)

            feats.append(feat.cpu().numpy())
            labels.append(target.numpy())

    feats = np.concatenate(feats, 0)
    labels = np.concatenate(labels, 0)

    prototype = []
    class_label = []
    for i in range(n_class):
        index = np.where(labels == i)[0]
        if len(index) > 0:
            class_label.append(int(i))
            feature_classwise = feats[index]
            prototype.append(torch.tensor(np.mean(feature_classwise, axis=0).reshape((1, -1))))
    #     ipdb.set_trace()
    return torch.cat(prototype, dim=0).cuda()


def gen_proto_global(feats, labels, n_classes):
    local_proto = []
    local_labels = []
    for i in range(n_classes):
        #         ipdb.set_trace()
        c_i = torch.nonzero(labels == i).reshape(-1)
        proto_i = torch.sum(feats[c_i], dim=0) / len(c_i)
        local_proto.append(proto_i.reshape(1, -1))
        local_labels.append(i)

    return torch.cat(local_proto, dim=0).cuda(), torch.tensor(local_labels).cuda()


import os
from PIL import Image, ImageFile
import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True  # 允许加载截断的图像

def build_BAR_client(n_env, dataroot, n_labels): # climbing:326/254  diving:520/426  fishing:163/162  pole vaulting:279/273  racing:336/334  throwing:317/317
    n_env = 3
    n_labels = 6
    mapping = {
    "climbing": 254, "diving": 426, "fishing": 162, "pole vaulting": 273, "racing":334, "throwing":317
}
    image_env = [[] for _ in range(n_env)]
    label_env = [[] for _ in range(n_env)]
    backimage_env = [[] for _ in range(n_env)]
    objimage_env = [[] for _ in range(n_env)]
    label_names = sorted([x for x in os.listdir(dataroot) if x != '.ipynb_checkpoints']) # climbing .....
    # ipdb.set_trace() # ['climbing', 'diving', 'fishing', 'pole vaulting', 'racing', 'throwing']
    for label_idx in range(n_labels): # 0-5
        folder_names = sorted([x for x in os.listdir(os.path.join(dataroot, label_names[label_idx])) if x != '.ipynb_checkpoints'])
        # folder_names = dino/BAR_Contextual/climbing 下的所有文件夹
        # ipdb.set_trace()
        print("train", label_idx, len(folder_names))  
        count = 0
        # ipdb.set_trace()
        while(count< mapping.get(label_names[label_idx])):
            for folder_name in folder_names: # 遍历下面的文件夹
                    path = os.path.join(dataroot, label_names[label_idx],folder_name)
                    file_names = os.listdir(path)
                    if len(file_names)==3:
                        for file_name in os.listdir(path): # file_name:only picture names
                    # ipdb.set_trace()     
                            file_path = os.path.join(path, file_name) 
                            # ipdb.set_trace()
                            try:
                                if file_name.startswith('1_'):
                                    objimag = Image.open(file_path)
                                    objimage_env[count%3].append(objimag)

                                elif file_name.startswith('2_'):
                                    backimag = Image.open(file_path)
                                    backimage_env[count%3].append(backimag)
                                    label_env[count%3].append(label_idx)
                                else:
                                    imag = Image.open(file_path)
                                    image_env[count%3].append(imag)

                            except (OSError, IOError) as e:
                                print(f"无法处理图像: {file_path}, 错误: {e}")
                        count = count + 1    
    labels = [np.array(client_label) for client_label in label_env]
#     ipdb.set_trace()
    for i in range(n_env):
        print(f"客户端 {i}: 原图数量={len(image_env[i])}, 主体图数量={len(objimage_env[i])}, 背景图数量={len(backimage_env[i])}, 标签数量={len(labels[i])}")  # 客户端0：590  客户端1：589  客户端2：587
    
    return image_env, objimage_env, backimage_env, labels                   
                                
                        

def build_client(n_env, dataroot, n_labels, mode='L7'):  # 7      9
    '''
    return:
        image_env: [(array,dtype=uint8)......]
        backimage_env: [(array,dtype=uint8)......]
        label_env: [(array)......]
    '''
    image_env = [[] for _ in range(n_env)]
    label_env = [[] for _ in range(n_env)]
    backimage_env = [[] for _ in range(n_env)]
    objimage_env = [[] for _ in range(n_env)]
    
    # 存储上下文环境（如背景、原图等的索引）
    env_cont = [range(10) for _ in range(n_labels)]
    
    # 获取类别标签目录
    label_names = sorted([x for x in os.listdir(dataroot) if x != '.ipynb_checkpoints'])
    

    if mode == 'L7':
        # 遍历类别和每个上下文
        for label_idx in range(n_labels): # 10个类，7个客户端，每个客户端有10个类中各一个背景。
            context_names = sorted([x for x in os.listdir(os.path.join(dataroot, label_names[label_idx])) if x != '.ipynb_checkpoints'])
            print("train", label_idx, len(context_names))
            for env_idx in range(len(context_names)-7, len(context_names)):  # n_env 0-6   2-8  背景
                print(env_idx)
                context_idx = env_cont[label_idx][env_idx]
    #             print(env_idx, env_cont[label_idx][env_idx])
                for folder_name in context_names:
                    if folder_name[0] == str(context_idx):
                        path = os.path.join(dataroot, label_names[label_idx], context_names[context_idx])
                        # ipdb.set_trace()
                        for folder_name1 in os.listdir(path):
                            folder_path = os.path.join(path, folder_name1)
                            file_names = os.listdir(folder_path)

                            if len(file_names) <= 1:
                                print(f"文件数量为 {len(file_names)}，跳过文件夹: {folder_path}")
                                continue

                            for file_name in file_names:
                                file_path = os.path.join(folder_path, file_name)
                                try:
                                    if file_name.startswith('1_'):
                                        backimag = Image.open(file_path)
                                        backimage_env[env_idx-(len(context_names)-7)].append(backimag)
                                        label_env[env_idx-(len(context_names)-7)].append(label_idx)
                                    elif file_name.startswith('2_'):
                                        objimag = Image.open(file_path)
                                        objimage_env[env_idx-(len(context_names)-7)].append(objimag)
                                    else:
                                        imag = Image.open(file_path)
                                        image_env[env_idx-(len(context_names)-7)].append(imag)
                                        if len(file_names) == 2:
                                            objimag = Image.open(file_path)
                                            objimage_env[env_idx-(len(context_names)-7)].append(objimag)
                                except (OSError, IOError) as e:
                                    print(f"无法处理图像: {file_path}, 错误: {e}")
                                    
    elif mode == 'F7':
        # 遍历类别和每个上下文
        for label_idx in range(n_labels):
            context_names = sorted([x for x in os.listdir(os.path.join(dataroot, label_names[label_idx])) if x != '.ipynb_checkpoints'])
            print("train", label_idx, context_names)
            for env_idx in range(n_env):
                context_idx = env_cont[label_idx][env_idx]
    #             print(env_idx, env_cont[label_idx][env_idx])
                for folder_name in context_names:
                    if folder_name[0] == str(context_idx):
                        path = os.path.join(dataroot, label_names[label_idx], context_names[context_idx])

                        for folder_name1 in os.listdir(path):
                            folder_path = os.path.join(path, folder_name1)
                            file_names = os.listdir(folder_path)

                            if len(file_names) <= 1:
                                print(f"文件数量为 {len(file_names)}，跳过文件夹: {folder_path}")
                                continue

                            for file_name in file_names:
                                file_path = os.path.join(folder_path, file_name)
                                try:
                                    if file_name.startswith('1_'):
                                        backimag = Image.open(file_path)
                                        backimage_env[env_idx].append(backimag)
                                        label_env[env_idx].append(label_idx)

                                    elif file_name.startswith('2_'):
                                        objimag = Image.open(file_path)
                                        objimage_env[env_idx].append(objimag)
                                    else:
                                        imag = Image.open(file_path)
                                        image_env[env_idx].append(imag)
                                        if len(file_names) == 2:
                                            objimag = Image.open(file_path)
                                            objimage_env[env_idx].append(objimag)

                                except (OSError, IOError) as e:
                                    print(f"无法处理图像: {file_path}, 错误: {e}")      

            
    labels = [np.array(client_label) for client_label in label_env]
#     ipdb.set_trace()
    for i in range(n_env):
        print(f"客户端 {i}: 原图数量={len(image_env[i])}, 主体图数量={len(objimage_env[i])}, 背景图数量={len(backimage_env[i])}, 标签数量={len(labels[i])}")
    
    return image_env, objimage_env, backimage_env, labels


def make_BAR_test(dataroot,transform):  # , transform
    all_image = []
    all_label = []
    mapping = {
    "climbing": 0, "diving": 1, "fishing": 2, "pole vaulting": 3, "racing":4, "throwing":5
}
    for file_name in os.listdir(dataroot):
        if '.ipynb_checkpoints' not in file_name:
            result = file_name.split('_')[0]
            file_path = os.path.join(dataroot,file_name)
            # ipdb.set_trace()
            temp=transform(Image.open(file_path).convert('RGB'))
            all_image.append(temp)
            all_label.append(mapping.get(result))
            # ipdb.set_trace()
    return all_image, all_label   


def make_test(dataroot, n_labels, n_context, n_env, transform, mode='L7'):  # 9  10  7
    all_image = []
    all_label = []
    all_context = []
    if n_env > n_context:
        print('Error: There are more environments than contexts.')
    label_names = os.listdir(dataroot)
    label_names = sorted([x for x in os.listdir(dataroot) if x != '.ipynb_checkpoints'])
    for label_idx in range(n_labels):
        context_names = os.listdir(dataroot +'/'+label_names[label_idx] + '/')
        context_names = sorted([x for x in context_names if x != '.ipynb_checkpoints'])
        print("test", label_idx, context_names)
#         ipdb.set_trace()
        if mode == 'F7':
            for context_idx in range(7, len(context_names)):  # n_env, len(context_names) |||  0, len(context_names)-7
                print(context_idx)
                path = dataroot +'/'+ label_names[label_idx] + '/' + context_names[context_idx] + '/'  # -len(context_names)
                print("test:"+path)
                image_names = os.listdir(path)
                image_names = sorted(image_names)
                for img in image_names:
                    if '.ipynb_checkpoints' not in img:
                        try:
                            temp=transform(Image.open(os.path.join(path, img)).convert('RGB'))
                            all_image.append(temp)
                            all_label.append(label_idx)
                            all_context.append(context_idx)
                        except IOError:
                            print('Warning: Broken file at ' + os.path.join(path, img))
        elif mode == 'L7':
            for context_idx in range(0, len(context_names)-7):  # n_env, len(context_names) |||  0, len(context_names)-7
                print(context_idx)
                path = dataroot +'/'+ label_names[label_idx] + '/' + context_names[context_idx] + '/'  # -len(context_names)
                print("test:"+path)
                image_names = os.listdir(path)
                image_names = sorted(image_names)
                for img in image_names:
                    if '.ipynb_checkpoints' not in img:
                        try:
                            temp=transform(Image.open(os.path.join(path, img)).convert('RGB'))
                            all_image.append(temp)
                            all_label.append(label_idx)
                            all_context.append(context_idx)
                        except IOError:
                            print('Warning: Broken file at ' + os.path.join(path, img))            
#     ipdb.set_trace()
    return all_image, all_label


class color_mnist_dataloader_1(torch.utils.data.Dataset): # MCGDM

        def __init__(self, features, back,labels, transform=None):
            self.features = features
            self.labels = labels
            self.backimg = back

        def __len__(self):
            return len(self.features)

        def __getitem__(self, idx):
            feature = self.features[idx]
            label = self.labels[idx]
            backimg = self.backimg[idx]

            return feature,feature, label

class color_mnist_dataloader(torch.utils.data.Dataset):

        def __init__(self, features, back,labels, transform=None):
            self.features = features
            self.labels = labels
            self.backimg = back

        def __len__(self):
            return len(self.features)

        def __getitem__(self, idx):
            feature = self.features[idx]
            label = self.labels[idx]
            backimg = self.backimg[idx]

            return feature,backimg, label
        

def get_color_mnist_dataloader(n_env):  # 10
    image_env = [[] for _ in range(n_env)]
    label_env = [[] for _ in range(n_env)]
    backimage_env = [[] for _ in range(n_env)]
    for i in range(n_env):
        loaded_data=torch.load('color_mnist_train_0.5/{}_color_data.pt'.format(i))
        for image in loaded_data:
            image_env[i].append(image)
        back_data = torch.load('color_mnist_train_0.5/{}_color_back.pt'.format(i))
        for back_img in back_data:
            backimage_env[i].append(back_img)
        train_labels=torch.load('color_mnist_train_0.5/{}_label.pt'.format(i))
        for label in train_labels:
            label_env[i].append(label.item())
        
    #print(loaded_data.size())
    # dataset=color_mnist_dataloader(loaded_data,back_data,train_labels)
    # train_dl=DataLoader(dataset,batch_size=64,shuffle=True)
    #test_dl=DataLoader(dataset_test,batch_size=64,shuffle=False)
    # test_dl = DataLoader(dataset_test, batch_size=1, shuffle=False)
    for i in range(n_env):
        # ipdb.set_trace()
        print(f"客户端 {i}: 原图数量={len(image_env[i])},  背景图数量={len(backimage_env[i])}, 标签数量={len(label_env[i])}")
        # ipdb.set_trace()
    return image_env,backimage_env,label_env



def get_color_mnist_test_dataloader(n_env):  # 10
    image_env = []
    label_env = []
    # backimage_env = [[] for _ in range(n_env)]
    
    loaded_data=torch.load('color_mnist_test/test_color_data.pt')
    for image in loaded_data:
            image_env.append(image)
            
    train_labels=torch.load('color_mnist_test/test_label.pt')
    for label in train_labels:
            label_env.append(label.item())
    label_env.append(train_labels)
    label_env = label_env[:-1]    
        
    #print(loaded_data.size())
    # dataset=color_mnist_dataloader(loaded_data,back_data,train_labels)
    # train_dl=DataLoader(dataset,batch_size=64,shuffle=True)
    #test_dl=DataLoader(dataset_test,batch_size=64,shuffle=False)
    # test_dl = DataLoader(dataset_test, batch_size=1, shuffle=False)
    print( f"原图数量={len(image_env)},   标签数量={len(label_env)}")
    return image_env,label_env

class color_mnist_test_dataloader(torch.utils.data.Dataset):

        def __init__(self, features, labels, transform=None):
            self.features = features
            self.labels = labels
            

        def __len__(self):
            return len(self.features)

        def __getitem__(self, idx):
            feature = self.features[idx]
            label = self.labels[idx]
            return feature, label


def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval() 

def gen_proto_local(net, dataloader, n_class=10, device='cuda:0'):
    feats = []
    labels = []
    net.eval()
    net.apply(fix_bn)
    net.to(device)
    with torch.no_grad():
        for batch_idx, (x, obj, back, target) in enumerate(dataloader):
            obj, target = obj.to(device), target.to(device)
            feat, _ = net(obj)

            feats.append(feat)
            labels.extend(target)

    feats = torch.cat(feats)
    labels = torch.tensor(labels)
#     ipdb.set_trace()
    prototype = []
    class_label = []
    for i in range(n_class):
        index = torch.nonzero(labels == i).reshape(-1)
        if len(index) > 0:
            class_label.append(int(i))
            feature_classwise = feats[index]
            prototype.append(torch.mean(feature_classwise, axis=0).reshape((1, -1)))
#     ipdb.set_trace()
    return torch.cat(prototype, dim=0), torch.tensor(class_label)

def gen_proto_global(feats, labels, n_classes):
    glo_protos = torch.zeros((n_classes, feats.shape[1]))
    for i in range(n_classes):
#         ipdb.set_trace()
        c_i = torch.nonzero(labels == i).reshape(-1)
        proto_i = torch.sum(feats[c_i], dim=0) / len(c_i)
        glo_protos[i] = proto_i
    
    return glo_protos





def insert_resized_image(base_image, insert_image):
    """
    将一张图像缩小到指定大小并随机插入到另一张图像的指定位置。
    
    参数:
        base_image_path (str): 第二张图像。
        insert_image_path (str): 要插入的第一张图像。
        insert_width (int): 缩小后的第一张图像的宽度。
        insert_height (int): 缩小后的第一张图像的高度。
        output_path (str, optional): 输出图像的文件路径。如果为 None，则仅显示图像，不保存。
    
    返回:
        Image: 处理后的图像对象。
    """
    # 获取底图的尺寸
    base_image = base_image.resize((224, 224))

    insert_width = random.choice([96, 128, 160])
    insert_height = random.choice([96, 128, 160])
    # 缩小插入的图像
    insert_image_resized = insert_image.resize((insert_width, insert_height))

    # 随机选择插入位置
    x = random.randint(0, 224 - insert_width)
    y = random.randint(0, 224 - insert_height)

    # 替换底图的对应区域
    base_image.paste(insert_image_resized, (x, y))
  
    return base_image



def blend_backgrounds_by_class(net_id, train_backimages, train_labels, group_count=3, output_path_prefix="output"):
    """
    将每个类别的背景图像分成指定数量的组，并将每组的图像融合成一个图像。
    
    参数:
        train_backimages (list of str): 所有背景图像的文件路径列表。
        train_labels (list of int): 与背景图像对应的类别标签列表。
        group_count (int): 每个类别的组数，默认是3。
        output_path_prefix (str): 输出图像的路径前缀。保存文件名会包含类别和组索引。
        
    返回:
        dict: 每个类别的融合图像路径字典 {label: [list of output paths]}。
    """
    # 将图像路径按类别进行分组
    class_images = defaultdict(list)
    for image_path, label in zip(train_backimages, train_labels):
        class_images[label].append(image_path)

    # 保存每个类别的融合图像路径
    output_paths = {}

    # 遍历每个类别
    for label, images in class_images.items():
        # 随机打乱图像并分组
        random.shuffle(images)
        groups = [images[i::group_count] for i in range(group_count)]

        # 融合每组图像
        blended_images = []
        for i, group in enumerate(groups):
            if not group:
                continue  # 跳过空组
            # 打开并转换图像为numpy数组
            image_arrays = [np.array(img) for img in group]
            # 计算平均值并生成融合图像
            blended_array = np.mean(image_arrays, axis=0).astype(np.uint8)
            blended_image = Image.fromarray(blended_array)
            mkdirs('./confounder_set/'+str(net_id))
            # 保存融合图像
            output_path = f"class{label}_group{i+1}.jpg"
            blended_image.save('./confounder_set/'+str(net_id)+'/'+output_path)



def build_confounds(dataset, mode):
    img_confounds=[]
    dataroot="./confounder_set/" + dataset + '/' + mode 
    for i in range(7):
        img_confounds.append([])
    clients = os.listdir(dataroot)
    clients = sorted([x for x in clients if x != '.ipynb_checkpoints'])
#     ipdb.set_trace()
#     print('build_confounds', clients)
    for client_idx in range(3):  # BAR:3  NICO:7
        client_elements = os.listdir(dataroot +'/'+ clients[client_idx] +'/')
        image_names = sorted([x for x in client_elements if x != '.ipynb_checkpoints'])

        path = dataroot +'/'+ clients[client_idx] + '/'
        
        for img in image_names:
            imag=Image.open(os.path.join(path,img))
            imag=imag.resize((224,224))
            img_confounds[client_idx].append(imag)

    return img_confounds













"""
if __name__ == '__main__':
   
    dataroot = "../Datasets/Vehicle"
    # 支持的图片扩展名
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}

# 计数器
    image_count = 0

# 遍历文件夹
    for root, dirs, files in os.walk(dataroot):
        for file in files:
        # 检查文件扩展名是否在支持的图片扩展名中
            if os.path.splitext(file)[1].lower() in image_extensions:
                image_count += 1

    print(f"总图片数量: {image_count}")
#     image_env, labels = build_client(7, "../Datasets/NICO", 10)
#     # image_env, backimage_env,labels = build_client2(7,'../dino/NICO_Contextual',10)

"""



def clip_feats_ext(clip_model, dataloader):
    
    name_classes = ["dog", "cat", "bear", "sheep", "bird", "rat", "horse", "elephant", "cow", "monkey"] 
    text_inputs = torch.cat([clip.tokenize(f"a photo of {c}.") for c in name_classes])
    
    clip_feats = torch.zeros((1000, 512))
    clip_logits = torch.zeros((1000, 10))
    clip_model = clip_model.float()
    
#     ipdb.set_trace()
    for batch_idx, (x, x_f, target, idx) in enumerate(dataloader): # //：结果向下取整到最接近的整数
    
        feats_clip, out_clip = clip_model(x, text_inputs)
        clip_feats[idx] = feats_clip.float()
        clip_logits[idx] = out_clip.float()

    return clip_feats, clip_logits
        


def gen_proto_clip(clip_model, dataloader, n_class=10):
    name_classes = ["dog", "cat", "bear", "sheep", "bird", "rat", "horse", "elephant", "cow", "monkey"] 
    text_inputs = torch.cat([clip.tokenize(f"a photo of {c}.") for c in name_classes])
    feats = []
    labels = []
    clip_model.eval()
    clip_model.cpu()
    with torch.no_grad():
        for batch_idx, (x, x_f, target, idx) in enumerate(dataloader):
            feat, _ = clip_model(x, text_inputs)

            feats.append(feat.cpu().numpy())
            labels.append(target.numpy())

    feats = np.concatenate(feats, 0)
    labels = np.concatenate(labels, 0)

    prototype = torch.zeros((10, feats.shape[1]))

    for i in range(n_class):
        index = np.where(labels == i)[0]
        if len(index) > 0:
            feature_classwise = feats[index]
            prototype[i] = torch.tensor(np.mean(feature_classwise, axis=0).reshape((1, -1)))
    #     ipdb.set_trace()
    return prototype.cuda()

def exp_lr_scheduler(optimizer, epoch, init_lr, lr_decay, decay_rate):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    # 每四次epoch调整一下lr，将lr减半
    lr = init_lr * (decay_rate ** (epoch // lr_decay))  # *是乘法，**是乘方，/是浮点除法，//是整数除法，%是取余数

    if epoch % lr_decay == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    # 返回改变了学习率的optimizer
    return optimizer

def count_directories(path):
    # 使用Path对象
    p = Path(path)
    # 统计文件夹数量
    folder_count = sum(1 for entry in p.iterdir() if entry.is_dir())
    return folder_count
if __name__ == '__main__':
    # make_BAR_test('../Datasets/BAR/test')
    # build_BAR_client(3,'../dino/BAR_Contextual',6)
    """
    count=0
    # 指定路径
    path = "../dino/BAR_Contextual/diving"
    for folder_name in os.listdir(path):
        folder_path = os.path.join(path,folder_name)
        # ipdb.set_trace()
        for file_name in os.listdir(folder_path):
            file_names = os.listdir(folder_path)
            # ipdb.set_trace()
            if len(file_names) == 3:
                count=count+1
    print(f"路径为 {path}，数量: {count}")
    """
    """
    # 统计文件夹数量
    num_folders = count_directories(path)
    print(f"路径 '{path}' 下的文件夹数量: {num_folders}")  # 326  fishing 163   pole vaulting:279  racing:336  throwing:317\
    """
    