import torch
from torch.utils.data import Dataset, DataLoader
import random
# import cv2
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from util import insert_resized_image
import os
from math import sqrt
# 普通的傅里叶变换
def colorful_spectrum_mix(img1, img2, alpha, ratio=1.0):
    """Input image size: ndarray of [H, W, C]"""
    lam = np.random.uniform(0, alpha)
    print(img1.shape, img2.shape)
    assert img1.shape == img2.shape
    h, w, c = img1.shape
    h_crop = int(h * sqrt(ratio))
    w_crop = int(w * sqrt(ratio))
    h_start = h // 2 - h_crop // 2
    w_start = w // 2 - w_crop // 2

    img1_fft = np.fft.fft2(img1, axes=(0, 1))
    img2_fft = np.fft.fft2(img2, axes=(0, 1))
    img1_abs, img1_pha = np.abs(img1_fft), np.angle(img1_fft)
    img2_abs, img2_pha = np.abs(img2_fft), np.angle(img2_fft)

    img1_abs = np.fft.fftshift(img1_abs, axes=(0, 1))
    img2_abs = np.fft.fftshift(img2_abs, axes=(0, 1))

    img1_abs_ = np.copy(img1_abs)
    img2_abs_ = np.copy(img2_abs)
    img1_abs[h_start:h_start + h_crop, w_start:w_start + w_crop] = \
        lam * img2_abs_[h_start:h_start + h_crop, w_start:w_start + w_crop] + (1 - lam) * img1_abs_[
                                                                                          h_start:h_start + h_crop,
                                                                                          w_start:w_start + w_crop]
    img2_abs[h_start:h_start + h_crop, w_start:w_start + w_crop] = \
        lam * img1_abs_[h_start:h_start + h_crop, w_start:w_start + w_crop] + (1 - lam) * img2_abs_[
                                                                                          h_start:h_start + h_crop,
                                                                                          w_start:w_start + w_crop]

    img1_abs = np.fft.ifftshift(img1_abs, axes=(0, 1))
    img2_abs = np.fft.ifftshift(img2_abs, axes=(0, 1))

    img21 = img1_abs * (np.e ** (1j * img1_pha))
    img12 = img2_abs * (np.e ** (1j * img2_pha))
    img21 = np.real(np.fft.ifft2(img21, axes=(0, 1)))
    img12 = np.real(np.fft.ifft2(img12, axes=(0, 1)))
    img21 = np.uint8(np.clip(img21, 0, 255))
    img12 = np.uint8(np.clip(img12, 0, 255))

    return img21, img12





class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return self.transform(x), self.transform(x)



class CustomNICO(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (str): 数据目录，包含每个类别的子文件夹。
            transform (callable, optional): 应用到图像上的变换函数。
        """
        self.data_dir = data_dir
        self.transform = transform 
        self.classes = sorted([x for x in os.listdir(data_dir) if x != '.ipynb_checkpoints']) # 获取所有类别的子文件夹
        self.class_to_idx = {cls_name[1:]: idx for idx, cls_name in enumerate(self.classes)}
        print(self.class_to_idx)
        self.samples = self._make_dataset()

    def _make_dataset(self):
        """
        遍历数据目录，收集所有图像文件路径及其标签。
        """
        samples = []
        for cls_name in self.classes:
            class_folder = os.path.join(self.data_dir, cls_name)
            if os.path.isdir(class_folder):
                for filename in os.listdir(class_folder):
                    img_path = os.path.join(class_folder, filename)
                    if img_path.endswith(('.jpg', '.jpeg', '.png')):
                        label = self.class_to_idx[cls_name[1:]]
                        samples.append((img_path, label))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')  # 打开图片并确保是RGB格式

        image = self.transform(image)

        return image, image, label, label


    
    
    
    
    

class NICO_dataset(torch.utils.data.Dataset):
    def __init__(self, all_data, all_label):
        super(NICO_dataset, self).__init__()
        self.all_data = all_data
        self.all_label = all_label


    def __getitem__(self, item):
        img = self.all_data[item]
        label = self.all_label[item]

        return img,label

    def __len__(self):
        return len(self.all_data)

class NICO_dataset_2(torch.utils.data.Dataset):
    def __init__(self, all_data, all_background, all_label):
        super(NICO_dataset_2, self).__init__()
        self.all_data = all_data
        self.all_background = all_background
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomCrop(224, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.52418953, 0.5233741, 0.44896784],
                                 std=[0.21851876, 0.2175944, 0.22552039])
        ])
        # self.all_sal=all_sal
        # self.all_confounds=all_confounds
        self.all_label = all_label

    def __getitem__(self, item):
        img = self.all_data[item]
        backimg = self.all_background[item]
        if img.mode != 'RGB':
            img = img.convert("RGB")
        if backimg.mode != 'RGB':
            backimg = backimg.convert("RGB")
        img = self.transform(img)
        backimg = self.transform(backimg)
        label = self.all_label[item]
        
        return img, backimg, label

    def __len__(self):
        return len(self.all_data)

class NICO_dataset_I(torch.utils.data.Dataset):
    def __init__(self, all_data, all_object, all_background, all_label):
        super(NICO_dataset_I, self).__init__()
        self.all_data = all_data
        self.all_object = all_object
        self.all_background = all_background
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomCrop(224, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.52418953, 0.5233741, 0.44896784],
                                 std=[0.21851876, 0.2175944, 0.22552039])
        ])
        # self.all_sal=all_sal
        # self.all_confounds=all_confounds
        self.all_label = all_label

    def __getitem__(self, item):
        img = self.all_data[item]
        objectimg = self.all_object[item]
        backimg = self.all_background[item]
        if img.mode != 'RGB':
            img = img.convert("RGB")
        if objectimg.mode != 'RGB':
            objectimg = objectimg.convert("RGB")
        if backimg.mode != 'RGB':
            backimg = backimg.convert("RGB")
        img = self.transform(img)
        objectimg = self.transform(objectimg)
        backimg = self.transform(backimg)
      
        label = self.all_label[item]
        
        return img, objectimg, backimg, label

    def __len__(self):
        return len(self.all_data)


class NICO_dataset_F(torch.utils.data.Dataset):
    def __init__(self, all_data, all_object, all_confounders, all_label):
        super(NICO_dataset_F, self).__init__()
        self.all_data = all_data
        self.all_object = all_object
        self.all_confounders = all_confounders
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomCrop(224, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.52418953, 0.5233741, 0.44896784],
                                 std=[0.21851876, 0.2175944, 0.22552039])
        ])
        # self.all_sal=all_sal
        # self.all_confounds=all_confounds
        self.all_label = all_label

    def __getitem__(self, item):
        img = self.all_data[item]
        objectimg = self.all_object[item]
        rand_back = random.choice(self.all_confounders)
        if (objectimg.size[0] <= 50 and objectimg.size[1] <= 50) or (objectimg.size[0] <= 30 or objectimg.size[1] <= 30):
            objectimg = img
#         img= img.resize((224,224))
#         print(img.size)
#         img = np.array(img)  
#         rand_back=np.array(rand_back)
#         imgF,imgF2 = colorful_spectrum_mix(img, rand_back, 0.5, ratio= 0.5)  
#         imgF = Image.fromarray(imgF)
#         imgF2 = Image.fromarray(imgF2)
#         imgF.save('output_img21.jpg')
#         imgF2.save('output_img22.jpg')
        
        fusion_img = insert_resized_image(rand_back, objectimg)
        if img.mode != 'RGB':
            img = img.convert("RGB")
        if objectimg.mode != 'RGB':
            objectimg = objectimg.convert("RGB")
        if fusion_img.mode != 'RGB':
            fusion_img = fusion_img.convert("RGB")  
        
        
        
        
        img = self.transform(img)
        objectimg = self.transform(objectimg)
        fusion_img = self.transform(fusion_img)
        label = self.all_label[item]
        
        return img, objectimg, fusion_img, label

    def __len__(self):
        return len(self.all_data) 
    
    
    
    
class NICO_dataset_3(torch.utils.data.Dataset):
    def __init__(self, all_data, all_label):
        super(NICO_dataset_3, self).__init__()
        self.all_data = all_data
        self.all_label = all_label
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomCrop(224, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.52418953, 0.5233741, 0.44896784],
                                 std=[0.21851876, 0.2175944, 0.22552039])
        ])

    def __getitem__(self, item):
        img = self.all_data[item]
        if img.mode != 'RGB':
            img = img.convert("RGB")
        img=self.transform(img)
        label = self.all_label[item]

        return img,label

    def __len__(self):
        return len(self.all_data)
    
class NICO_dataset_4(torch.utils.data.Dataset):
    def __init__(self, all_data,all_label):
        super(NICO_dataset_4, self).__init__()
        self.all_data = all_data
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomCrop(224, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.52418953, 0.5233741, 0.44896784],
                                 std=[0.21851876, 0.2175944, 0.22552039])
        ])
        # self.all_sal=all_sal
        # self.all_confounds=all_confounds
        self.all_label = all_label
        # self.to_pil = transforms.ToPILImage()
    def __getitem__(self, item):
        img = self.all_data[item]
        if img.mode != 'RGB':
            img = img.convert("RGB")
        # sal = self.all_sal[item]
        # confound=random.sample(self.all_confounds,1)
        # final_img=self.transform(concat_img_1(img,sal,confound))
        img1 = self.transform(img)
        # img1_pil = self.to_pil(img1)
        img2 = self.transform(img)
        label = self.all_label[item]
        label = torch.tensor(label)

        return img1,img2,label

    def __len__(self):
        return len(self.all_data)
    
class NICO_MCGDM(torch.utils.data.Dataset):
    def __init__(self, all_data, all_label):
        super(NICO_MCGDM, self).__init__()
        self.all_data = all_data
        self.all_label = all_label
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomCrop(224, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.52418953, 0.5233741, 0.44896784],
                                 std=[0.21851876, 0.2175944, 0.22552039])
        ])

        

    def __getitem__(self, item):
        img = self.all_data[item]
        data1 = self.transform(img)
        data2 = self.transform(img)
        label = self.all_label[item]

        return data1, data2, label

    def __len__(self):
        return len(self.all_data)
        

def get_NICO_dataloader_train(id):
    seed = 1000
    torch.manual_seed(seed)
    random.seed(seed)
    #original data
    loaded_data = torch.load('NICO2/nico_client{}.pt'.format(id+1))
    loaded_label=torch.load('NICO2/nico_client{}_label.pt'.format(id+1))
    #frame data
    frame_data = torch.load('NICO_frame/nico_client{}.pt'.format(id + 1))
    frame_label = torch.load('NICO_frame/nico_client{}_label.pt'.format(id + 1))


    dataset=NICO_dataset(loaded_data,frame_data,loaded_label)

    train_dl=DataLoader(dataset,batch_size=64,shuffle=True)


    return train_dl

def get_NICO_dataloader_test():
    seed = 1000
    torch.manual_seed(seed)
    random.seed(seed)
    #labels
    test_data=torch.load('NICO_test/nico_test.pt')
    test_labels=torch.load('NICO_test/nico_test_label.pt')
    dataset_test=NICO_dataset_2(test_data,test_labels)
    test_dl=DataLoader(dataset_test,batch_size=64,shuffle=True)

    return test_dl

'''def concat_img(image_path, sal_image_path, context_image_path):

    # 定义目标尺寸
    target_size = (224, 224)

    # 加载二值黑白图（掩膜），并调整大小
    bw_mask = cv2.imread(sal_image_path, cv2.IMREAD_GRAYSCALE)
    bw_mask = cv2.resize(bw_mask, target_size)  # 调整大小

    # 确保掩膜是二值的（0或255）
    _, bw_mask = cv2.threshold(bw_mask, 127, 255, cv2.THRESH_BINARY)

    # 加载两张彩色图像，并调整大小
    color_image1 = cv2.imread(image_path)
    color_image1 = cv2.resize(color_image1, target_size)  # 调整大小

    color_image2 = cv2.imread(context_image_path)
    color_image2 = cv2.resize(color_image2, target_size)  # 调整大小

    # 使用掩膜提取第一张彩色图像中的特定部分
    bw_mask_color = cv2.cvtColor(bw_mask, cv2.COLOR_GRAY2BGR)  # 将掩膜转换为彩色
    extracted_part = cv2.bitwise_and(color_image1, bw_mask_color)

    # 在第二张彩色图像上应用提取的部分
    inverse_mask = cv2.bitwise_not(bw_mask_color)  # 创建与掩膜相反的掩膜
    background = cv2.bitwise_and(color_image2, inverse_mask)  # 保留背景
    final_image = cv2.add(background, extracted_part)  # 结合背景和提取的部分
    cv2.imshow('Final Image', final_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return final_image'''


def concat_img_1(image, sal_image, context_image):
    #输出是PIL

    # 加载二值黑白图（掩膜），并调整大小finished
    #bw_mask = Image.open(sal_image_path).convert('L')
    #bw_mask = bw_mask.resize(target_size)

    # 确保掩膜是二值的（0或255）
    bw_mask = np.array(sal_image.point(lambda x: 0 if x < 128 else 255), dtype=np.uint8)

    # 加载两张彩色图像，并调整大小
    #color_image1 = Image.open(image_path).resize(target_size)
    #color_image2 = Image.open(context_image_path).resize(target_size)

    # 使用掩膜提取第一张彩色图像中的特定部分
    extracted_part = np.array(image) * (bw_mask[:, :, None] / 255)

    # 在第二张彩色图像上应用提取的部分
    inverse_mask = 1 - bw_mask / 255

    background = np.array(context_image[0]) * (inverse_mask[:, :, None])

    # 结合背景和提取的部分
    final_image_array = np.uint8(extracted_part + background)

    # 将图像数组转换回PIL图像
    final_image = Image.fromarray(final_image_array)


    return final_image