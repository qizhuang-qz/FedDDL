import os
import sys
import json
import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.ops import box_convert
from datasets import TinyImageNet_load
import ipdb

# 设置 Grounding DINO 路径
sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))
from GroundingDINO.groundingdino.util.inference import load_model, predict

# 设备配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 类别名（Tiny-ImageNet 示例）
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

CIFAR100_CLASSES = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']

# 归一化 & 反归一化设置
channel_mean = (0.4914, 0.4822, 0.4465)
channel_std = (0.2470, 0.2435, 0.2616)
MEAN = [-m / s for m, s in zip(channel_mean, channel_std)]
STD = [1 / s for s in channel_std]
denormalizer = transforms.Normalize(mean=MEAN, std=STD)
normalize = transforms.Normalize(mean=channel_mean, std=channel_std)

# 加载模型
CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
CHECKPOINT_PATH = "weight/groundingdino_swint_ogc.pth"
groundingdino_model = load_model(CONFIG_PATH, CHECKPOINT_PATH, device=DEVICE)


# ========= 提取器类 =========
class Patcher:
    def __init__(self):
        self.bbox_dict = {}  # key: idx (int), value: xyxy bbox list

    def __call__(self, idx, image, label, prompt, model, max_region=224):
        BOX_THRESHOLD = 0.15
        TEXT_THRESHOLD = 0.15

        # 去归一化并转 uint8
        image = denormalizer(image).clamp(0, 1)
        image_np = image.permute(1, 2, 0).numpy() * 255
        image_np = image_np.astype(np.uint8)

        # resize 到模型输入大小
        image_np = cv2.resize(image_np, (max_region, max_region))
        image_tensor = transforms.ToTensor()(Image.fromarray(image_np))

        # Grounding DINO 预测
        boxes, logits, phrases = predict(
            model=model,
            image=image_tensor,
            caption=prompt,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            device=DEVICE,
        )

        if boxes is None or len(boxes) == 0:
            return

        # 取得分最高的 bbox
        _, indices = torch.sort(logits, descending=True)
        top_idx = indices[0].item()
        box = boxes[top_idx]

        h, w, _ = image_np.shape
        box = box * torch.tensor([w, h, w, h])
        xyxy = box_convert(box.unsqueeze(0), in_fmt="cxcywh", out_fmt="xyxy").squeeze(0).tolist()
        xyxy = [round(float(x), 2) for x in xyxy]
        
        # 原始尺寸
        original_w, original_h = 224, 224
        target_w, target_h = 32, 32
        scale_x = target_w / original_w
        scale_y = target_h / original_h

        # 转换为目标图尺寸的 bbox
        xyxy_32 = [round(float(x) * scale_x, 2) if i % 2 == 0 else round(float(x) * scale_y, 2)
                   for i, x in enumerate(xyxy)]

        # 保存的是缩放后的 bbox
        self.bbox_dict[str(idx)] = xyxy_32



# ========= 主程序 =========
if __name__ == "__main__":
    SAVE_JSON = "cifar100_bbox_idx_only.json"
    patcher = Patcher()

    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
#     dataset = TinyImageNet_load(root='../datasets/tiny-imagenet-200/', train=True, transform=transform)
    dataset = CIFAR100(root='../datasets', train=True, download=True, transform=transform)
    ipdb.set_trace()
    for idx in tqdm(range(len(dataset))):
        image, label = dataset[idx]
        prompt = CIFAR100_CLASSES[label]
        patcher(idx, image, label, prompt, groundingdino_model)

    # 保存为 JSON 文件（idx → bbox）
    with open(SAVE_JSON, "w") as f:
        json.dump(patcher.bbox_dict, f, indent=2)



# import os
# import sys
# import cv2
# import numpy as np
# from PIL import Image
# from tqdm import tqdm
# import torch
# from torchvision import transforms
# from torchvision.datasets import CIFAR10, CIFAR100
# from torchvision.ops import box_convert
# from datasets import TinyImageNet_load

# # Grounding DINO
# sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))
# from GroundingDINO.groundingdino.util.inference import load_model, predict

# # 配置设备
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # CIFAR-10 类别
# CIFAR10_CLASSES = [
#     'airplane', 'automobile', 'bird', 'cat', 'deer',
#     'dog', 'frog', 'horse', 'ship', 'truck'
# ]

# CIFAR100_CLASSES = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']

# Tiny_CLASSES = [
#     'goldfish', 'European fire salamander', 'bullfrog', 'tailed frog',
#     'American alligator', 'boa constrictor', 'trilobite', 'scorpion',
#     'black widow', 'tarantula', 'centipede', 'goose', 'koala', 'jellyfish',
#     'brain coral', 'snail', 'slug', 'sea slug', 'American lobster',
#     'spiny lobster', 'black stork', 'king penguin', 'albatross', 'dugong',
#     'Chihuahua', 'Yorkshire terrier', 'golden retriever', 'Labrador retriever',
#     'German shepherd', 'standard poodle', 'tabby', 'Persian cat', 'Egyptian cat',
#     'cougar', 'lion', 'brown bear', 'ladybug', 'fly', 'bee', 'grasshopper',
#     'walking stick', 'cockroach', 'mantis', 'dragonfly', 'monarch',
#     'sulphur butterfly', 'sea cucumber', 'guinea pig', 'hog', 'ox', 'bison',
#     'bighorn', 'gazelle', 'Arabian camel', 'orangutan', 'chimpanzee', 'baboon',
#     'African elephant', 'lesser panda', 'abacus', 'academic gown', 'altar',
#     'apron', 'backpack', 'bannister', 'barbershop', 'barn', 'barrel',
#     'basketball', 'bathtub', 'beach wagon', 'beacon', 'beaker', 'beer bottle',
#     'bikini', 'binoculars', 'birdhouse', 'bow tie', 'brass', 'broom', 'bucket',
#     'bullet train', 'butcher shop', 'candle', 'cannon', 'cardigan',
#     'cash machine', 'CD player', 'chain', 'chest', 'Christmas stocking',
#     'cliff dwelling', 'computer keyboard', 'confectionery', 'convertible',
#     'crane', 'dam', 'desk', 'dining table', 'drumstick', 'dumbbell',
#     'flagpole', 'fountain', 'freight car', 'frying pan', 'fur coat', 'gasmask',
#     'go-kart', 'gondola', 'hourglass', 'iPod', 'jinrikisha', 'kimono',
#     'lampshade', 'lawn mower', 'lifeboat', 'limousine', 'magnetic compass',
#     'maypole', 'military uniform', 'miniskirt', 'moving van', 'nail',
#     'neck brace', 'obelisk', 'oboe', 'organ', 'parking meter', 'pay-phone',
#     'picket fence', 'pill bottle', 'plunger', 'pole', 'police van', 'poncho',
#     'pop bottle', "potter's wheel", 'projectile', 'punching bag', 'reel',
#     'refrigerator', 'remote control', 'rocking chair', 'rugby ball', 'sandal',
#     'school bus', 'scoreboard', 'sewing machine', 'snorkel', 'sock',
#     'sombrero', 'space heater', 'spider web', 'sports car',
#     'steel arch bridge', 'stopwatch', 'sunglasses', 'suspension bridge',
#     'swimming trunks', 'syringe', 'teapot', 'teddy', 'thatch', 'torch',
#     'tractor', 'triumphal arch', 'trolleybus', 'turnstile', 'umbrella',
#     'vestment', 'viaduct', 'volleyball', 'water jug', 'water tower', 'wok',
#     'wooden spoon', 'comic book', 'plate', 'guacamole', 'ice cream',
#     'ice lolly', 'pretzel', 'mashed potato', 'cauliflower', 'bell pepper',
#     'mushroom', 'orange', 'lemon', 'banana', 'pomegranate', 'meat loaf',
#     'pizza', 'potpie', 'espresso', 'alp', 'cliff', 'coral reef', 'lakeside',
#     'seashore', 'acorn'
# ]

# # 均值与标准差
# channel_mean = (0.4914, 0.4822, 0.4465)
# channel_std = (0.2470, 0.2435, 0.2616)

# # 图像归一化与反归一化
# MEAN = [-mean / std for mean, std in zip(channel_mean, channel_std)]
# STD = [1 / std for std in channel_std]
# denormalizer = transforms.Normalize(mean=MEAN, std=STD)
# normalize = transforms.Normalize(mean=channel_mean, std=channel_std)

# # Grounding DINO 模型加载
# CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
# CHECKPOINT_PATH = "weight/groundingdino_swint_ogc.pth"
# groundingdino_model = load_model(CONFIG_PATH, CHECKPOINT_PATH, device=DEVICE)


# def extract_bbox_region(image, bbox_ls, logits, phrases, max_len=1):
#     _, indices = torch.sort(logits, descending=True)
#     indices = indices[:max_len].tolist()
#     bbox_ls = bbox_ls[indices]
#     phrases = [phrases[idx] for idx in indices]

#     h, w, _ = image.shape
#     extracted_regions = []
#     valid_phrases = []
#     for bbox, phrase in zip(bbox_ls, phrases):
#         bbox = bbox * torch.Tensor([w, h, w, h])
#         xyxy = box_convert(boxes=bbox, in_fmt="cxcywh", out_fmt="xyxy").numpy()
#         x1, y1, x2, y2 = xyxy.astype(int)
#         bbox_region = image[y1:y2, x1:x2]
#         if bbox_region.shape[0] == 0 or bbox_region.shape[1] == 0:
#             continue
#         extracted_regions.append(bbox_region)
#         valid_phrases.append(phrase)
#     return extracted_regions, valid_phrases


# class Patcher:
#     def __init__(self, save_root):
#         self.save_root = save_root
#         os.makedirs(self.save_root, exist_ok=True)

#     def __call__(self, image, label, prompt, f_name, model, max_region=224):
#         BOX_THRESHOLD = 0.15
#         TEXT_THRESHOLD = 0.15

#         # 图像反归一化并转换为 uint8
#         image = denormalizer(image).clamp(0, 1)
#         image_np = image.permute(1, 2, 0).numpy() * 255
#         image_np = image_np.astype(np.uint8)

#         # Resize 图像
#         image_np = cv2.resize(image_np, (max_region, max_region))
#         image_tensor = transforms.ToTensor()(Image.fromarray(image_np))

#         # Grounding DINO 预测
#         boxes, logits, phrases = predict(
#             model=model,
#             image=image_tensor,
#             caption=prompt,
#             box_threshold=BOX_THRESHOLD,
#             text_threshold=TEXT_THRESHOLD,
#             device=DEVICE,
#         )

#         class_name = Tiny_CLASSES[label]
#         class_dir = os.path.join(self.save_root, class_name)
#         os.makedirs(class_dir, exist_ok=True)

#         img_id = os.path.splitext(os.path.basename(f_name))[0]
#         output_dir = os.path.join(class_dir, img_id)
#         os.makedirs(output_dir, exist_ok=True)

#         # 保存原图
#         cv2.imwrite(os.path.join(output_dir, "original.jpg"), cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

#         # 提取前景区域
#         regions, phrases = extract_bbox_region(image_np, boxes, logits, phrases)
#         mask = np.zeros(image_np.shape[:2], dtype=np.uint8)

#         for i, (region, phrase) in enumerate(zip(regions, phrases), start=1):
#             bbox = boxes[i-1] * torch.tensor([max_region, max_region, max_region, max_region])
#             xyxy = box_convert(boxes=bbox.unsqueeze(0), in_fmt="cxcywh", out_fmt="xyxy").numpy().astype(int)[0]
#             x1, y1, x2, y2 = np.clip(xyxy, 0, max_region)

#             mask[y1:y2, x1:x2] = 255  # 累积前景掩码
# #             region = cv2.resize(region, (max_region, max_region))
#             out_path = os.path.join(output_dir, f"{i}_{phrase}.jpg")
#             cv2.imwrite(out_path, cv2.cvtColor(region, cv2.COLOR_RGB2BGR))

#         # 保存背景图像
#         inv_mask = cv2.bitwise_not(mask)
#         background = cv2.bitwise_and(image_np, image_np, mask=inv_mask)
#         cv2.imwrite(os.path.join(output_dir, "2_background.jpg"), cv2.cvtColor(background, cv2.COLOR_RGB2BGR))


# if __name__ == "__main__":
#     SAVE_PATH = "CIFAR10_Contextual"
#     patcher = Patcher(save_root=SAVE_PATH)

#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         normalize,
#     ])
#     dataset = TinyImageNet_load(root='../datasets/tiny-imagenet-200/', train=True, transform=transform)
# #     dataset = CIFAR100(root='../datasets', train=True, download=True, transform=transform)
#     for idx in tqdm(range(len(dataset))):
#         image, label = dataset[idx]
#         prompt = Tiny_CLASSES[label]
#         filename = f"{prompt}_{idx}.jpg"
#         patcher(image, label, prompt, filename, groundingdino_model)  