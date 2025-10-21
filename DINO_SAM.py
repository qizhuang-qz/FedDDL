import os
import sys
import cv2
sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))

import os
import ipdb
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import re
from einops import rearrange
# Grounding DINO
from GroundingDINO.groundingdino.util import box_ops
# ipdb.set_trace()
from GroundingDINO.groundingdino.util.inference import load_model
from GroundingDINO.groundingdino.util.inference import predict
from GroundingDINO.groundingdino.util.inference import load_image_new
from GroundingDINO.groundingdino.util.inference import annotate
from torchvision.ops import box_convert
from nico_tied import get_NICO_dataloader_train

# segment anything
# from segment_anything.segment_anything import build_sam, SamPredictor
from segment_anything.segment_anything import build_sam, SamPredictor
# import build_sam
# from predictor import SamPredictor
import cv2
import numpy as np

# diffusers
import torch

import shutil
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# from config import get_train_config
#
# config = get_train_config()
# mean=[0.52418953, 0.5233741, 0.44896784],
# std=[0.21851876, 0.2175944, 0.22552039]
# 归一化,反归一化
channel_mean = (0.52418953, 0.5233741, 0.44896784)
channel_std = (0.21851876, 0.2175944, 0.22552039)
MEAN = [-mean / std for mean, std in zip(channel_mean, channel_std)]
STD = [1 / std for std in channel_std]
denormalizer = transforms.Normalize(mean=MEAN, std=STD)
normalize = transforms.Normalize(mean=channel_mean, std=channel_std)

DEVICE = torch.device("cuda")
# Load Grounding DINO model

# ——————————————————————————集中处理 节省时间 防止多次加载------------------------------------------------------

CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
CHECKPOINT_PATH = "weight/groundingdino_swint_ogc.pth"
groundingdino_model = load_model(CONFIG_PATH, CHECKPOINT_PATH, device=DEVICE) # 权重

# sam_checkpoint = 'weight/sam_vit_h_4b8939.pth'
# sam = build_sam(checkpoint=sam_checkpoint)
# # sam.cuda()
# # sam.to(device)
# sam_predictor = SamPredictor(sam)


def show_mask(mask, image, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
        # color = np.array([0,0,0,1])
    color = torch.Tensor(color).cuda()  # 新增
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    annotated_frame_pil = Image.fromarray(image).convert("RGBA")
    mask_image_pil = Image.fromarray((mask_image.cpu().numpy() * 255).astype(np.uint8)).convert("RGBA")
    return np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil))


def MMA(m):
    for i in range(m.shape[0] - 1):
        m[i + 1][0] = m[i][0] | m[i + 1][0]
    m = m[-1][0].unsqueeze(0).unsqueeze(0)
    return m


def fore_filter(mask, image):
    mask = mask.repeat(3, 1, 1).cpu().numpy().astype(np.uint8)
    mask = np.transpose(mask, (1, 2, 0))
    image = image[..., ::-1]  # RGB -> BGR
    foreground_x = mask * image
    return foreground_x


def back_filter(mask, image):
    mask = ~mask
    mask = mask.repeat(3, 1, 1).cpu().numpy().astype(np.uint8)
    mask = np.transpose(mask, (1, 2, 0))
    image = image[..., ::-1]
    background_x = mask * image
    return background_x


class DINOSAM():
    def __init__(self, save_path):
        self.save_path = save_path

    def __call__(self, image, label, content, f_name, groundingdino_model, w):
        # Run Grounding DINO for detection
        BOX_TRESHOLD = 0.15   # 0,25
        TEXT_TRESHOLD = 0.15
        SCORES = []
        Fx = []
        Bx = []
        save_path = self.save_path
        TEXT_PROMPT = content
        SCORES_temp = []
        Fx_temp = []
        Bx_temp = []
        # ipdb.set_trace()
        f_name_1 = f_name  #  ../Datasets/NICO/0dog/0on grass/1.jpg  ../Datasets/BAR/train/climbing_0.jpg
        folder_name = f_name_1.split('/')[-2]  # 0on grass
        image_path = f_name_1 # f_name # os.path.join(f'/Vireo172_Image/ready_chinese_food/{label + 1}', f_name)
        # ipdb.set_trace()
        f_name = f_name_1.split('/')[-1]  # f_name[0].split('/')[-1]  单个图片
        # ipdb.set_trace()
        # ipdb.set_trace() climbing_0.jpg   label=0
        if not os.path.exists(os.path.join(save_path, '{}'.format(label))):
            os.makedirs(os.path.join(save_path, '{}'.format(label)))
            
        # category_path = '{}/{}'.format(save_path, label)  # 0
        # if not os.path.exists(os.path.join(category_path, '{}'.format(folder_name))):
          #   os.makedirs(os.path.join(category_path, '{}'.format(folder_name)))  # 0/0 on grass
        
        category_path_1 = '{}/{}'.format(save_path, label)  # 0/0 on grass  category_path, folder_name
        if not os.path.exists(os.path.join(category_path_1, '{}'.format(f_name))):
            os.makedirs(os.path.join(category_path_1, '{}'.format(f_name))) # 0/0 on grass/1.jpg
        rfb_path = '{}/{}'.format(category_path_1, f_name)
        # ipdb.set_trace()
        
        image_source = cv2.imread(image_path)  # .convert("RGB") # .convert("RGB")  # Image.open  cv2.imread
        image_source = cv2.cvtColor(image_source, cv2.COLOR_BGR2RGB)
        image_source = cv2.resize(image_source, (224,224))
        image_source = np.asarray(image_source)
        H, W, _ = image_source.shape

        boxes, logits, phrases = predict(
            model=groundingdino_model,
            image=image,
            caption=TEXT_PROMPT,
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD,
            device=DEVICE,
        )
        
#         ipdb.set_trace()
        
        extracted_regions, phrases = extract_bbox_region(image_source, boxes, logits, phrases)

        shutil.copy(image_path, rfb_path)
        assert len(extracted_regions) == len(phrases), "unequal length"
        for i, (image, phrase) in enumerate(zip(extracted_regions, phrases), start=1):
            image = cv2.resize(image, (224,224))
            try:
                cv2.imwrite(f'{rfb_path}/{i}_{phrase}.jpg', image[..., ::-1])
            except:
                print('empty image!')
#                 import ipdb
#                 ipdb.set_trace()
                   
        
#         annotated = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)  # BGR的格式
#         annotated = annotated[..., ::-1]  # BGR to RGB
#         Image.fromarray(annotated)
#         cv2.imwrite(f"{rfb_path}/annotated_{f_name}.jpg", annotated[:, :, ::-1])

#         if boxes.numel() == 0:
#             # masks = torch.zeros(1,1,240,320,dtype=torch.bool).cuda() # msrvtt 240 320 h w
#             masks = torch.zeros(1, 1, H, W, dtype=torch.bool)  # .cuda()  # activity .cuda()
#             score = torch.zeros(1)  # .to(device="cpu")
#         else:
#             sam_predictor.set_image(image_source)
        
#             boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
#             transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image_source.shape[:2]).cuda()
#             masks, scores, _ = sam_predictor.predict_torch(
#                 point_coords=None,
#                 point_labels=None,
#                 boxes=transformed_boxes,
#                 multimask_output=False,
#             )
            # score = max(scores)  # 取最大得分
            # masks = MMA(masks)  # MutiMask 1 1 224 224
        # 策略一
        #         foreground_x = fore_filter(masks[0][0],image_source) # numpy
        #         background_x = back_filter(masks[0][0],image_source)
        # 策略二
        # foreground_x = show_mask(masks[0][0], image_source[..., ::-1])  # numpy
        # masks = ~masks
        # background_x = show_mask(masks[0][0], image_source[..., ::-1])

        # save all masks
#         for idx in range(len(phrases)):
#             out = show_mask(masks[idx][0], image_source[..., ::-1])
#             cv2.imwrite(f"{rfb_path}/{phrases[idx]}.jpg", out)

#         # 保留原始图像，防止失真
#         shutil.copy(image_path, rfb_path)
#         try:
#             os.rename(rfb_path + r'/' + f_name, rfb_path + r'/' + '0.jpg')
#         except FileExistsError:
#             print('file exists.')

#         # cv2.imwrite(r"{}/1.jpg".format(rfb_path), foreground_x)
#         # cv2.imwrite(r"{}/2.jpg".format(rfb_path), background_x)

        # w.write(f'{label + 1}/{f_name}, {phrases}, {content}\n')
    
def extract_bbox_region(image, bbox_ls, logits, phrases, max_len=1):
    _, indices = torch.sort(logits,descending=True)
    indices = indices[:max_len].tolist()
    bbox_ls = bbox_ls[indices]
    phrases = [phrases[idx] for idx in indices]
    
    h, w,_ = image.shape

    extracted_regions = []
    valid_phrases = [] 
    for bbox, phrase in zip(bbox_ls, phrases):   # for bbox in bbox_ls:
        
        bbox = bbox * torch.Tensor([w, h, w, h])
        xyxy = box_convert(boxes=bbox, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        
        x1, y1, x2, y2 = xyxy.astype(int)
#         assert x1 != x2 and y1!= y2, f"{x1, y1, x2, y2}, bbox: {bbox}"
        """
        x1 = max(0, min(255, x1))
        x2 = max(0, min(255, x2))
        y1 = max(0, min(255, y1))
        y2 = max(0, min(255, y2))
        
        if x1 == x2:
            x2 = x1 + 5
        if y1 == y2:
            y2 = y2 + 5
        """
        """
        # 确保 x1 < x2 和 y1 < y2
        if x1 >= x2 or y1 >= y2:
            print(f"Invalid bounding box: {(x1, y1, x2, y2)}, skipping...")
            continue  # 跳过无效的边界框
        """
        bbox_region = image[y1:y2, x1:x2] # y1=197 y2=367  x1=52 x2=190
        
        if bbox_region.shape[0] == 0 or bbox_region.shape[1] == 0:
            print(f"Skipping empty bbox_region with shape: {bbox_region.shape}, coordinates: {(x1, y1, x2, y2)}")
            continue  # 跳过无效的 bbox_region
            
        # ipdb.set_trace()  # (171, 138, 3)
        assert 0 not in bbox_region.shape, f"wrong shape! {bbox_region.shape, x1, y1, x2, y2}"
        
        extracted_regions.append(bbox_region)
        valid_phrases.append(phrase)

    return extracted_regions, valid_phrases

CONTENT = 'Scrambled egg.Water.Fried flour.Streaky pork slices.Spareribs.Chicken chunks.Sweet and sour sauce.Shredded pork.Sweet dumplings.Coconut cake.Rice dumpling.Minced pork.Beef chunks.Fried bread stick.Spare ribs chunks.Gluten.Millet.Steamed Bun.Scallion pancake.Tofu chunks.Bitter gourd slices.Chili oil.Fish.Dumplings.Chicken Wings.Boiled chicken slices.Batonnet tenderloin.Chicken Feet.Sweet potato starch noodles.Crab.shumai.Tentacles of Squid .Duck neck.Spring rolls.Rice.Mashed potatoes.Steamed twisted roll.Crushed hot and dry chili.Crayfish.Fresh Shrimp.Rice noodle roll.Noodles.Suckling pig.Glutinous rice.Boild eggs.Crushed garlic.Stinky tofu.Garlic bulb.Pork slices.Hob blocks of potato.Bullfrog.Hob blocks of eggplant.Pork leg.Shanghai cabbage.Lotus root box.Vermicelli.Chinese cabbage.Whole chicken.Ribbonfish.Chicken legs.Preserved egg chunks .Pancakes.Brunoise diced chicken.Beef slices.Steamed egg custard.Whole green pepper.Crystal sugar.Chinese Kale.Water spinach.Lettuce.Chili saurce.White yam.blueberry jam.Double-side fried egg.Garlic sprout pieces.Oyster sauce.Small loaf of steamed bread.Beans.Okra slices.Egg cake.Pork chunks.Fish slices.Green beans.Groundnut kernels.Green soy beans.Hot and dry pepper powder.Shredded chicken.Loofah.Shredded beef tripe.Preserved vegetables.Clams.Mutton slices.River snail.Yam chunks .Pork intestines.Celery stalk slices.Steamed Rice Powder.Lotus root slices.Minced green onion.Pepper slices.Pork paste.Dried mushroom.Cabbage.Tenderloin slices.Spaghetti.White onion.Shredded potato.Sliced Fatty Beef.Chives.Brunoise diced carrot.Meat balls.Sea cucumber.Brunoise diced cucumber.Yuba.Fern root noodles.Minced pickled beans.Crushed pepper.Black fungus.Eggplant.Egg drop.Shredded kelp.Duck head.Lentinus edodes.Garlic clove.Red dates.White congee.barbecued pork slices.Corn kernels.Snow peas.Sausage slices.Dried pieces of bean curd.Celtuce sclices.Boild egg slices.Wonton.Shredded pepper.Crushed preserved egg.Shelled fresh shrimps.Pine nuts.Cold steamed rice noodles.Chive pieces.Celery.Crucian.Black chicken chunks.Steak.Black sesame'

CONTENT = "cabbage.beef.white fungus.carrot.mango.carrot.shrimp.bacon"
image1=cv2.imread("../Datasets/NICO/0dog/0on grass/0.jpg")
height, width = image1.shape[:2]
target_size = (width, height)

if __name__ == '__main__':
    import ipdb
    import torch.nn as nn
    import random

    stage = "train"
    dinosam = DINOSAM("NICO_Contextual_2025_4_8")  # f"./{stage.upper()}_REGIONS_BY_PREDICTING_TOP5_ANOTHER"

    # code for single image
#     image = Image.open('/Vireo172_Image/ready_chinese_food/120/4_8.jpg').convert('RGB')
#     transform = transforms.Compose([
#     transforms.Resize(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225])
#     ])
#     image = transform(image)
#     with open('test_examples.txt', 'a', encoding='utf-8') as w:
#         dinosam(image, 119, CONTENT, ['/120/4_8.jpg'], groundingdino_model, w)
#         exit(0)
 
    # from data import build_dataset
    # loader = get_NICO_dataloader_train(0)
#     ipdb.set_trace()
    # image1=cv2.imread("../Datasets/NICO/0dog/0on grass/0.jpg")
    # height, width = image1.shape[:2]
    # target_size = (width, height)
    with open(f'regions_{stage}_by_predicting_top7_another.txt', 'w', encoding='utf-8') as w:
        
        base_dir =  '../Datasets/NICO/0dog'
        for root, dirs, files in os.walk(base_dir): # 遍历子文件夹
            for file in files:
            # 检查文件扩展名，确保是图片文件
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                # 获取图片的完整路径
                    image_path = os.path.join(root, file)
                    
                    image_name=image_path.split('/')[-1]
                    # ipdb.set_trace()
                    if image_name.startswith("throwing"):
                        # ipdb.set_trace()
                        label = 0
                        content='dog'
                # 使用 cv2.imread 读取图片
                        image = cv2.imread(image_path)
                    # image = image[0]
                        if image is None:
                            print("Image not found or unable to load.")
                            continue
                        resized_img = cv2.resize(image, (224,224))
                   
                        dinosam(resized_img, label, content, image_path , groundingdino_model, w)
                
        
         
        """
        image=cv2.imread("../Datasets/NICO/0dog/0on grass/0.jpg")
        image1=cv2.imread("../Datasets/NICO/0dog/6running/96.jpg")
        height, width = image.shape[:2]
        target_size = (width, height)
        resized_img = cv2.resize(image1, target_size)
        # ipdb.set_trace()
        content='dog'   # 'climbing'
        label=0
        f_name="../Datasets/NICO/0dog/6running/96.jpg"   # "../Datasets/BAR/train/climbing_106.jpg ../Datasets/NICO/0dog/0on grass/1.jpg"
        dinosam(resized_img, label, content, f_name, groundingdino_model, w)
        
        for bid, batch_data in enumerate(loader):  # batch_size=1  每次1张   tqdm(enumerate(loader, start=1))
            image, label, content, f_name = batch_data
            label = label.item()
#             content = CONTENT
#             word = content.split('.')
#             random.shuffle(word)
#             content = ".".join(word)
            image = image[0]
            content = content[0]
            dinosam(image, label, content, f_name, groundingdino_model, w)
            """
            
