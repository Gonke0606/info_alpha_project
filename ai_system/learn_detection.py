"""
物体検出モデル（RetinaNet）の学習と評価
"""
import os
import json
import random
import copy
from collections import deque
from typing import Callable, Sequence, Tuple, Union, Dict, List
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, sampler, ConcatDataset
import torchvision.transforms.functional as TF
import torchvision.transforms as TVT
from torchvision.ops import sigmoid_focal_loss, batched_nms
import numpy as np
from PIL import Image
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from models_detection import RetinaNet, convert_to_xywh, convert_to_xyxy


def get_dataset_statistics(dataset: Dataset):
    def custom_collate_fn_for_stats(batch):
        images = [item[0] for item in batch if item is not None and item[0] is not None]
        if not images:
            return None, None
        return torch.stack(images, 0), None

    data_loader = DataLoader(
        dataset,
        batch_size=64,
        num_workers=0,
        shuffle=False,
        collate_fn=custom_collate_fn_for_stats
    )
    
    mean = torch.zeros(3)
    std = torch.zeros(3)
    num_pixels = 0
    
    for images, _ in tqdm(data_loader, desc="[Calculating Stats]"):
        if images is None:
            continue
            
        batch_size, num_channels, height, width = images.shape
        num_pixels += batch_size * height * width
        mean += images.sum(dim=(0, 2, 3))
        std += (images ** 2).sum(dim=(0, 2, 3))
        
    mean /= num_pixels
    std = torch.sqrt(std / num_pixels - mean ** 2)
    
    return mean, std


def generate_subset(dataset: ConcatDataset, ratio: float, random_seed: int = 0):
    num_total = len(dataset)
    indices = list(range(num_total))
    size = int(num_total * ratio)
    
    random.seed(random_seed)
    random.shuffle(indices)
    
    val_indices = indices[:size]
    train_indices = indices[size:]
    
    return train_indices, val_indices


def calc_iou(boxes1: torch.Tensor, boxes2: torch.Tensor):
    intersect_left_top = torch.maximum(boxes1[:, :2].unsqueeze(1), boxes2[:, :2])
    intersect_right_bottom = torch.minimum(boxes1[:, 2:].unsqueeze(1), boxes2[:, 2:])
    intersect_width_height = (intersect_right_bottom - intersect_left_top).clamp(min=0)
    
    intersect_areas = intersect_width_height.prod(dim=2)
    
    areas1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    areas2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    union_areas = areas1.unsqueeze(1) + areas2 - intersect_areas
    
    return intersect_areas / union_areas.clamp(min=1e-8), union_areas


class VisionTransformWrapper:
    def __init__(self, transform: Callable):
        self.transform = transform
        
    def __call__(self, img: Image, target: dict):
        return self.transform(img), target


class ResizeForStats:
    def __init__(self, size):
        self.size = size
        
    def __call__(self, img, target):
        return TF.resize(img, self.size), target


class RandomHorizontalFlip:
    def __init__(self, prob: float = 0.5):
        self.prob = prob
        
    def __call__(self, img: Image, target: dict):
        if random.random() < self.prob and target['boxes'].numel() > 0:
            img = TF.hflip(img)
            target['boxes'][:, [0, 2]] = img.size[0] - target['boxes'][:, [2, 0]]
            
        return img, target


class RandomSizeCrop:
    def __init__(self, scale: Sequence[float], ratio: Sequence[float]):
        self.scale = scale
        self.ratio = ratio
        
    def __call__(self, img: Image, target: dict):
        width, height = img.size
        top, left, h, w = TVT.RandomResizedCrop.get_params(img, self.scale, self.ratio)
        
        img = TF.crop(img, top, left, h, w)
        
        if target['boxes'].numel() > 0:
            target['boxes'][:, ::2] = (target['boxes'][:, ::2] - left).clamp(min=0, max=w)
            target['boxes'][:, 1::2] = (target['boxes'][:, 1::2] - top).clamp(min=0, max=h)
            
            keep = (target['boxes'][:, 2] > target['boxes'][:, 0]) & (target['boxes'][:, 3] > target['boxes'][:, 1])
            
            if 'classes' in target:
                target['classes'] = target['classes'][keep]
            target['boxes'] = target['boxes'][keep]
            
        target['size'] = torch.tensor([w, h], dtype=torch.int64)
        
        return img, target


class RandomResize:
    def __init__(self, min_sizes: Sequence[int], max_size: int):
        self.min_sizes = min_sizes
        self.max_size = max_size
        
    def _get_target_size(self, w, h, target_min_size):
        min_s, max_s = (w, h) if w < h else (h, w)
        
        if max_s / min_s * target_min_size > self.max_size:
            target_min_size = int(self.max_size * min_s / max_s)
            
        resized_w = int(target_min_size * w / min_s)
        resized_h = int(target_min_size * h / min_s)
        
        return resized_w, resized_h
        
    def __call__(self, img, target):
        min_size = random.choice(self.min_sizes)
        w, h = img.size
        
        resized_w, resized_h = self._get_target_size(w, h, min_size)
        
        img = TF.resize(img, (resized_h, resized_w))
        
        if target['boxes'].numel() > 0:
            target['boxes'] *= resized_w / w
            
        target['size'] = torch.tensor([resized_w, resized_h], dtype=torch.int64)
        
        return img, target


class ToTensor:
    def __call__(self, img, target):
        return TF.to_tensor(img), target


class Normalize:
    def __init__(self, mean: Sequence[float], std: Sequence[float]):
        self.mean = mean
        self.std = std
        
    def __call__(self, img, target):
        return TF.normalize(img, mean=self.mean, std=self.std), target


class Compose:
    def __init__(self, transforms: Sequence[Callable]):
        self.transforms = transforms
        
    def __call__(self, img: Image, target: dict):
        for transform in self.transforms:
            img, target = transform(img, target)
        return img, target


class RandomSelect:
    def __init__(self, t1: Callable, t2: Callable, p: float = 0.5):
        self.t1 = t1
        self.t2 = t2
        self.p = p
        
    def __call__(self, img: Image, target: dict):
        if random.random() < self.p:
            return self.t1(img, target)
        else:
            return self.t2(img, target)


class CocoFormatDataset(Dataset):
    def __init__(self, root_dir: str, ann_file: str, global_class_map: Dict[str, int], transform=None, dataset_idx: int = 0):
        self.root_dir = root_dir
        self.transform = transform
        self.coco = COCO(ann_file)
        self.global_class_map = global_class_map
        self.dataset_idx = dataset_idx
        
        local_cat_ids = self.coco.getCatIds()
        self.target_cat_ids = [cid for cid in local_cat_ids if self.coco.loadCats(cid)[0]['name'] in self.global_class_map]
        
        valid_image_ids = set()
        if 'annotations' in self.coco.dataset:
            for ann in self.coco.dataset['annotations']:
                if ann.get('category_id') in self.target_cat_ids:
                    valid_image_ids.add(ann['image_id'])
                    
        self.ids = [img_id for img_id in sorted(self.coco.imgs.keys()) if img_id in valid_image_ids]
        
        self.local_cat_id_to_global_label = {
            cid: self.global_class_map[self.coco.loadCats(cid)[0]['name']] for cid in self.target_cat_ids
        }

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx: int):
        coco_img_id = self.ids[idx]
        img_info = self.coco.loadImgs(coco_img_id)[0]
        img_path = os.path.join(self.root_dir, img_info['file_name'])
        
        try:
            img = Image.open(img_path).convert('RGB')
        except (IOError, OSError):
            return None, None
            
        w, h = img.size
        orig_size = torch.tensor([w, h], dtype=torch.int64)
        
        ann_ids = self.coco.getAnnIds(imgIds=coco_img_id, catIds=self.target_cat_ids)
        anns = self.coco.loadAnns(ann_ids)
        
        boxes = []
        labels = []
        
        for ann in anns:
            if ann.get('category_id') not in self.local_cat_id_to_global_label:
                continue
                
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(self.local_cat_id_to_global_label[ann['category_id']])
            
        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32)
        else:
            boxes = torch.empty((0, 4), dtype=torch.float32)
        
        if labels:
            labels = torch.tensor(labels, dtype=torch.int64)
        else:
            labels = torch.empty(0, dtype=torch.int64)
            
        target = {
            'boxes': boxes,
            'classes': labels,
            'image_id': torch.tensor([coco_img_id]),
            'orig_size': orig_size,
            'file_name': img_info['file_name'],
            'dataset_idx': self.dataset_idx
        }
        
        if self.transform:
            img, target = self.transform(img, target)
            
        return img, target


def post_process(preds_class, preds_box, anchors, targets, conf_threshold=0.05, nms_threshold=0.5):
    anchors_xywh = convert_to_xywh(anchors)
    preds_box[:, :, :2] = anchors_xywh[:, :2] + preds_box[:, :, :2] * anchors_xywh[:, 2:]
    preds_box[:, :, 2:] = preds_box[:, :, 2:].exp() * anchors_xywh[:, 2:]
    preds_box = convert_to_xyxy(preds_box)
    
    preds_class = preds_class.sigmoid()
    
    scores = []
    labels = []
    boxes = []
    
    for img_preds_class, img_preds_box, img_targets in zip(preds_class, preds_box, targets):
        img_preds_box[:, ::2] = img_preds_box[:, ::2].clamp(min=0, max=img_targets['size'][0])
        img_preds_box[:, 1::2] = img_preds_box[:, 1::2].clamp(min=0, max=img_targets['size'][1])
        
        if 'orig_size' in img_targets and img_targets['size'][0] > 0:
            scale_factor = img_targets['orig_size'][0].item() / img_targets['size'][0].item()
            img_preds_box *= scale_factor
            
        img_preds_score, img_preds_label = img_preds_class.max(dim=1)
        
        keep = img_preds_score > conf_threshold
        img_preds_score = img_preds_score[keep]
        img_preds_label = img_preds_label[keep]
        img_preds_box = img_preds_box[keep]
        
        keep_indices = batched_nms(img_preds_box, img_preds_score, img_preds_label, nms_threshold)
        
        scores.append(img_preds_score[keep_indices])
        labels.append(img_preds_label[keep_indices])
        boxes.append(img_preds_box[keep_indices])
        
    return scores, labels, boxes


def loss_func(preds_class, preds_box, anchors, targets, iou_lower_threshold=0.4, iou_upper_threshold=0.5):
    anchors_xywh = convert_to_xywh(anchors)
    loss_class = preds_class.new_tensor(0)
    loss_box = preds_class.new_tensor(0)
    
    for img_preds_class, img_preds_box, img_targets in zip(preds_class, preds_box, targets):
        if img_targets['classes'].shape[0] == 0:
            targets_class = torch.zeros_like(img_preds_class)
            loss_class += sigmoid_focal_loss(img_preds_class, targets_class, reduction='sum')
            continue
            
        ious, _ = calc_iou(anchors.to(img_targets['boxes'].device), img_targets['boxes'])
        ious_max, ious_argmax = ious.max(dim=1)
        
        targets_class = torch.full_like(img_preds_class, -1)
        targets_class[ious_max < iou_lower_threshold] = 0
        
        positive_masks = ious_max > iou_upper_threshold
        num_positive_anchors = positive_masks.sum()
        
        targets_class[positive_masks] = 0
        assigned_classes = img_targets['classes'][ious_argmax]
        targets_class[positive_masks, assigned_classes[positive_masks]] = 1
        
        loss_class += ((targets_class != -1) * sigmoid_focal_loss(img_preds_class, targets_class, reduction='none')).sum() / num_positive_anchors.clamp(min=1)
        
        if num_positive_anchors == 0:
            continue
            
        assigned_boxes = img_targets['boxes'][ious_argmax]
        assigned_boxes_xywh = convert_to_xywh(assigned_boxes)
        
        targets_box = torch.zeros_like(img_preds_box)
        targets_box[:, :2] = (assigned_boxes_xywh[:, :2] - anchors_xywh[:, :2]) / anchors_xywh[:, 2:]
        targets_box[:, 2:] = (assigned_boxes_xywh[:, 2:] / anchors_xywh[:, 2:]).log()
        
        loss_box += F.smooth_l1_loss(img_preds_box[positive_masks], targets_box[positive_masks], beta=1/9)
        
    batch_size = preds_class.shape[0]
    loss_class /= batch_size
    loss_box /= batch_size
    
    return loss_class, loss_box


def collate_func(batch):
    batch = [b for b in batch if b[0] is not None]
    if not batch:
        return None, None
        
    max_height = 0
    max_width = 0
    
    for img, _ in batch:
        height, width = img.shape[1:]
        max_height = max(max_height, height)
        max_width = max(max_width, width)
        
    height = (max_height + 31) // 32 * 32
    width = (max_width + 31) // 32 * 32
    
    imgs = batch[0][0].new_zeros((len(batch), 3, height, width))
    targets = []
    
    for i, (img, target) in enumerate(batch):
        h, w = img.shape[1:]
        imgs[i, :, :h, :w] = img
        targets.append(target)
        
    return imgs, targets
    

def evaluate(data_loader: DataLoader, model: nn.Module, loss_func: Callable, global_class_map: Dict[str, int], conf_threshold=0.05, nms_threshold=0.5):
    model.eval()
    losses = []
    coco_preds = []
    
    merged_gt_data = {
        'info': {},
        'licenses': [],
        'images': [],
        'annotations': [],
        'categories': []
    }
    
    for name, index in global_class_map.items():
        merged_gt_data['categories'].append({'id': index, 'name': name, 'supercategory': 'all'})
        
    img_id_counter = 0
    ann_id_counter = 0
    img_id_map = {}
    
    if hasattr(data_loader.sampler, 'indices'):
        val_indices = data_loader.sampler.indices
    else:
        val_indices = range(len(data_loader.dataset))
    
    concat_dataset = data_loader.dataset
    
    for global_idx in val_indices:
        dataset_idx = np.searchsorted(concat_dataset.cumulative_sizes, global_idx, side='right')
        
        if dataset_idx > 0:
            local_idx = global_idx - concat_dataset.cumulative_sizes[dataset_idx-1]
        else:
            local_idx = global_idx
        
        original_dataset = concat_dataset.datasets[dataset_idx]
        original_coco_img_id = original_dataset.ids[local_idx]
        
        img_info = original_dataset.coco.loadImgs(original_coco_img_id)[0]
        
        new_img_id = img_id_counter
        img_id_map[(original_dataset.dataset_idx, original_coco_img_id)] = new_img_id
        
        new_img_info = img_info.copy()
        new_img_info['id'] = new_img_id
        merged_gt_data['images'].append(new_img_info)
        
        ann_ids = original_dataset.coco.getAnnIds(imgIds=original_coco_img_id)
        anns = original_dataset.coco.loadAnns(ann_ids)
        
        for ann in anns:
            cat_name = original_dataset.coco.loadCats(ann['category_id'])[0]['name']
            if cat_name in global_class_map:
                new_ann = ann.copy()
                new_ann['id'] = ann_id_counter
                new_ann['image_id'] = new_img_id
                new_ann['category_id'] = global_class_map[cat_name]
                merged_gt_data['annotations'].append(new_ann)
                
                ann_id_counter += 1
                
        img_id_counter += 1
        
    coco_gt = COCO()
    coco_gt.dataset = merged_gt_data
    coco_gt.createIndex()
    
    for imgs, targets in tqdm(data_loader, desc='[Validation]'):
        if imgs is None or not targets:
            continue
            
        with torch.no_grad():
            imgs = imgs.to(model.get_device())
            targets_on_device = [{k: v.to(model.get_device()) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
            
            preds_class, preds_box, anchors = model(imgs)
            loss_class, loss_box = loss_func(preds_class, preds_box, anchors, targets_on_device)
            
            loss = loss_class + loss_box
            losses.append(loss.item())
            
            scores, labels, boxes = post_process(preds_class, preds_box, anchors, targets_on_device, conf_threshold, nms_threshold)
            
            for i in range(len(targets)):
                original_coco_img_id = targets[i]['image_id'].item()
                dataset_idx = targets[i]['dataset_idx']
                
                new_img_id = img_id_map.get((dataset_idx, original_coco_img_id))
                if new_img_id is None:
                    continue
                    
                for score, label, box in zip(scores[i], labels[i], boxes[i]):
                    coco_box = [box[0].item(), box[1].item(), (box[2] - box[0]).item(), (box[3] - box[1]).item()]
                    coco_preds.append({
                        'image_id': new_img_id,
                        'category_id': label.item(),
                        'score': score.item(),
                        'bbox': coco_box
                    })

    if losses:
        val_loss = np.mean(losses)
    else:
        val_loss = float('inf')
        
    print(f'\nValidation loss = {val_loss:.4f}')

    if not coco_preds:
        return val_loss
        
    coco_dt = coco_gt.loadRes(coco_preds)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    
    coco_eval.params.imgIds = list(img_id_map.values())
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    return val_loss


def visualize_predictions(model, data_loader, device, num_images=5, conf_threshold=0.3):
    model.eval()
    
    try:
        imgs, targets = next(iter(data_loader))
    except StopIteration:
        return
        
    if imgs is None or not targets:
        return
        
    imgs = imgs.to(device)
    targets_on_device = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
    
    with torch.no_grad():
        preds_class, preds_box, anchors = model(imgs)
        scores, labels, boxes = post_process(preds_class, preds_box, anchors, targets_on_device, conf_threshold)
        
    num_to_show = min(num_images, len(imgs))
    concat_dataset = data_loader.dataset
    global_class_map = concat_dataset.datasets[0].global_class_map
    class_names = sorted(global_class_map.keys(), key=lambda k: global_class_map[k])
    
    for i in range(num_to_show):
        target_info = targets[i]
        dataset_idx = target_info['dataset_idx']
        
        original_dataset = concat_dataset.datasets[dataset_idx]
        data_root = original_dataset.root_dir
        
        original_img_path = os.path.join(data_root, target_info['file_name'])
        img_pil = Image.open(original_img_path).convert('RGB')
        
        fig, ax = plt.subplots(1, figsize=(12, 9))
        ax.imshow(img_pil)
        
        for score, label_idx, box in zip(scores[i], labels[i], boxes[i]):
            xmin, ymin, xmax, ymax = box.cpu()
            width = xmax - xmin
            height = ymax - ymin
            
            class_name = class_names[label_idx.item()]
            
            rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            
            plt.text(xmin, ymin, f'{class_name}: {score.item():.2f}', bbox=dict(facecolor='yellow', alpha=0.5))
            
        ax.set_title(f"Image '{target_info['file_name']}' Detections")
        plt.axis('off')
        plt.show()

class ConfigTrainEval:
    def __init__(self):
        self.datasets = [
            {'root': '/Users/wakabayashikengo/fatigue_detection/DataFacePose', 'ann': '/Users/wakabayashikengo/fatigue_detection/DataFacePose/_annotations.coco.json'}
        ]
        self.save_file = '/Users/wakabayashikengo/face_pose.pth'
        self.stats_file = '/Users/wakabayashikengo/face_pose.json'
        
        self.val_ratio = 0.2
        self.num_epochs = 100
        self.lr_drop = [60, 80]
        self.val_interval = 5
        self.lr = 1e-4
        self.clip = 0.1
        self.moving_avg = 100
        self.batch_size = 4
        self.num_workers = 0
        
        if torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'


def train_eval():
    config = ConfigTrainEval()
    
    min_sizes = (608, 640, 672, 704, 736)
    
    all_class_names = set()
    for ds_info in config.datasets:
        try:
            coco = COCO(ds_info['ann'])
            cats = coco.loadCats(coco.getCatIds())
            for cat in cats:
                all_class_names.add(cat['name'])
        except Exception as e:
            pass

    global_class_list = sorted(list(all_class_names))
    global_class_map = {name: i for i, name in enumerate(global_class_list)}
    num_classes = len(global_class_list)
    
    transforms_for_stats = Compose([ResizeForStats((512, 512)), ToTensor()])
    stats_datasets = []
    for i, ds_info in enumerate(config.datasets):
        dataset = CocoFormatDataset(
            root_dir=ds_info['root'],
            ann_file=ds_info['ann'],
            global_class_map=global_class_map,
            transform=transforms_for_stats,
            dataset_idx=i
        )
        stats_datasets.append(dataset)
        
    full_stats_dataset = ConcatDataset(stats_datasets)
    
    channel_mean, channel_std = get_dataset_statistics(full_stats_dataset)
    print(f"データセットの平均値と標準偏差を計算: mean={channel_mean.tolist()}, std={channel_std.tolist()}")
    
    stats = {'mean': channel_mean.tolist(), 'std': channel_std.tolist()}
    os.makedirs(os.path.dirname(config.stats_file), exist_ok=True)
    with open(config.stats_file, 'w') as f:
        json.dump(stats, f, indent=4)

    train_transforms = Compose([
        RandomHorizontalFlip(), 
        VisionTransformWrapper(TVT.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2)),
        RandomSelect(
            RandomResize(min_sizes, max_size=1333),
            Compose([
                RandomSizeCrop(scale=(0.8, 1.0), ratio=(0.75, 1.333)), 
                RandomResize(min_sizes, max_size=1333),
            ])
        ),
        ToTensor(), 
        Normalize(mean=channel_mean, std=channel_std),
    ])
    
    test_transforms = Compose([
        RandomResize((min_sizes[-1],), max_size=1333), 
        ToTensor(), 
        Normalize(mean=channel_mean, std=channel_std),
    ])

    train_datasets = []
    for i, ds_info in enumerate(config.datasets):
        dataset = CocoFormatDataset(
            root_dir=ds_info['root'],
            ann_file=ds_info['ann'],
            global_class_map=global_class_map,
            transform=train_transforms,
            dataset_idx=i
        )
        train_datasets.append(dataset)
    
    full_dataset = ConcatDataset(train_datasets)
    
    train_indices, val_indices = generate_subset(full_dataset, config.val_ratio)

    train_sampler = sampler.SubsetRandomSampler(train_indices)
    val_sampler = sampler.SubsetRandomSampler(val_indices)

    val_datasets_list = []
    for i, ds_info in enumerate(config.datasets):
        dataset = CocoFormatDataset(
            root_dir=ds_info['root'],
            ann_file=ds_info['ann'],
            global_class_map=global_class_map,
            transform=test_transforms,
            dataset_idx=i
        )
        val_datasets_list.append(dataset)

    full_val_dataset = ConcatDataset(val_datasets_list)

    train_loader = DataLoader(
        full_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        sampler=train_sampler,
        collate_fn=collate_func
    )
    
    val_loader = DataLoader(
        full_val_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        sampler=val_sampler,
        collate_fn=collate_func
    )

    model = RetinaNet(num_classes)
    model.backbone.load_state_dict(torch.hub.load_state_dict_from_url('https://download.pytorch.org/models/resnet18-5c106cde.pth'), strict=False)
    model.to(config.device)
    
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.lr_drop, gamma=0.1)
    
    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(config.num_epochs):
        model.train()
        losses_class = deque()
        losses_box = deque()
        losses = deque()
        
        with tqdm(train_loader) as pbar:
            pbar.set_description(f'[エポック {epoch + 1}]')
            for imgs, targets in pbar:
                if imgs is None or not targets:
                    continue
                    
                imgs = imgs.to(config.device)
                targets = [{k: v.to(config.device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
                
                optimizer.zero_grad()
                
                preds_class, preds_box, anchors = model(imgs)
                loss_class, loss_box = loss_func(preds_class, preds_box, anchors, targets)
                loss = loss_class + loss_box
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip)
                optimizer.step()
                
                losses_class.append(loss_class.item())
                losses_box.append(loss_box.item())
                losses.append(loss.item())
                
                if len(losses) > config.moving_avg:
                    losses_class.popleft()
                    losses_box.popleft()
                    losses.popleft()
                    
                pbar.set_postfix({
                    'loss': f'{np.mean(losses):.3f}',
                    'cls_loss': f'{np.mean(losses_class):.3f}',
                    'box_loss': f'{np.mean(losses_box):.3f}'
                })
        
        scheduler.step()
        
        if (epoch + 1) % config.val_interval == 0:
            val_loss = evaluate(val_loader, model, loss_func, global_class_map)
            if val_loss < best_val_loss:
                print(f"  -> 検証ロスが改善しました ({best_val_loss:.4f} -> {val_loss:.4f})。ベストモデルを更新します。")
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(model.state_dict())
            else:
                print(f"  -> 検証ロスは改善しませんでした (best: {best_val_loss:.4f})。")
    
    if best_model_state is not None:
        final_save_path = config.save_file.replace('.pth', '_best.pth')
        os.makedirs(os.path.dirname(final_save_path), exist_ok=True)
        torch.save(best_model_state, final_save_path)
        model.load_state_dict(best_model_state)
    else:
        final_save_path = config.save_file.replace('.pth', '_final.pth')
        torch.save(model.state_dict(), final_save_path)

    visualize_predictions(model, val_loader, config.device, num_images=5)

def main():
    train_eval()

if __name__ == "__main__":
    main()