"""
物体検出タスクで使用するモデルアーキテクチャとユーティリティ関数を定義するモジュール
"""
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.ops.misc import FrozenBatchNorm2d
from torchvision.ops import batched_nms
from typing import Tuple, List

def convert_to_xywh(boxes: torch.Tensor) -> torch.Tensor:
    wh = boxes[..., 2:] - boxes[..., :2]
    xy = boxes[..., :2] + wh / 2
    return torch.cat((xy, wh), dim=-1)

def convert_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    xymin = boxes[..., :2] - boxes[..., 2:] / 2
    xymax = boxes[..., 2:] + xymin
    return torch.cat((xymin, xymax), dim=-1)

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = FrozenBatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = FrozenBatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride > 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                FrozenBatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x); out = self.bn1(out); out = self.relu(out)
        out = self.conv2(out); out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity; out = self.relu(out)
        return out

class ResNet18Backbone(nn.Module):
    """バックボーンネットワークのResNet18"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = FrozenBatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = nn.Sequential(BasicBlock(64, 64), BasicBlock(64, 64))
        self.layer2 = nn.Sequential(BasicBlock(64, 128, stride=2), BasicBlock(128, 128))
        self.layer3 = nn.Sequential(BasicBlock(128, 256, stride=2), BasicBlock(256, 256))
        self.layer4 = nn.Sequential(BasicBlock(256, 512, stride=2), BasicBlock(512, 512))
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.max_pool(x)
        x = self.layer1(x)
        x = self.dropout(x)
        c3 = self.layer2(x)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return c3, c4, c5

class FeaturePyramidNetwork(nn.Module):
    """特徴ピラミッドネットワーク"""
    def __init__(self, num_features: int=256):
        super().__init__()
        self.levels = (3, 4, 5, 6, 7)
        self.p6 = nn.Conv2d(512, num_features, kernel_size=3, stride=2, padding=1)
        self.p7_relu = nn.ReLU(inplace=True)
        self.p7 = nn.Conv2d(num_features, num_features, kernel_size=3, stride=2, padding=1)
        self.p5_1 = nn.Conv2d(512, num_features, kernel_size=1)
        self.p5_2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.p4_1 = nn.Conv2d(256, num_features, kernel_size=1)
        self.p4_2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.p3_1 = nn.Conv2d(128, num_features, kernel_size=1)
        self.p3_2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)

    def forward(self, c3, c4, c5):
        p6 = self.p6(c5); p7 = self.p7(self.p7_relu(p6))
        p5 = self.p5_1(c5); p5_up = F.interpolate(p5, size=c4.shape[-2:])
        p5 = self.p5_2(p5)
        p4 = self.p4_1(c4) + p5_up; p4_up = F.interpolate(p4, size=c3.shape[-2:])
        p4 = self.p4_2(p4)
        p3 = self.p3_1(c3) + p4_up; p3 = self.p3_2(p3)
        return p3, p4, p5, p6, p7

class DetectionHead(nn.Module):
    """検出ヘッド"""
    def __init__(self, num_channels_per_anchor: int, num_anchors: int=9, num_features: int=256):
        super().__init__()
        self.num_anchors = num_anchors
        self.conv_blocks = nn.ModuleList([
            nn.Sequential(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.ReLU(inplace=True))
            for _ in range(4)
        ])
        self.out_conv = nn.Conv2d(num_features, num_anchors * num_channels_per_anchor, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor):
        for block in self.conv_blocks:
            x = block(x)
        x = self.out_conv(x)
        bs, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(bs, w * h * self.num_anchors, -1)
        return x

class AnchorGenerator:
    def __init__(self, levels: tuple):
        ratios = torch.tensor([0.5, 1.0, 2.0])
        scales = torch.tensor([2 ** 0, 2 ** (1 / 3), 2 ** (2 / 3)])
        self.num_anchors = ratios.shape[0] * scales.shape[0]
        self.strides = [2 ** level for level in levels]
        self.anchors = []
        for level in levels:
            base_length = 2 ** (level + 2)
            scaled_lengths = base_length * scales
            anchor_areas = scaled_lengths ** 2
            anchor_widths = (anchor_areas / ratios.unsqueeze(1)) ** 0.5
            anchor_heights = anchor_widths * ratios.unsqueeze(1)
            anchor_widths, anchor_heights = anchor_widths.flatten(), anchor_heights.flatten()
            self.anchors.append(torch.stack((-0.5 * anchor_widths, -0.5 * anchor_heights, 0.5 * anchor_widths, 0.5 * anchor_heights), dim=1))

    @torch.no_grad()
    def generate(self, feature_sizes: List[Tuple[int, int]], device: torch.device) -> torch.Tensor:
        all_anchors = []
        for stride, level_anchors, feature_size in zip(self.strides, self.anchors, feature_sizes):
            height, width = feature_size
            xs = (torch.arange(width, device=device) + 0.5) * stride
            ys = (torch.arange(height, device=device) + 0.5) * stride
            grid_x, grid_y = torch.meshgrid(xs, ys, indexing='xy')
            grid_x, grid_y = grid_x.flatten(), grid_y.flatten()
            
            level_anchors = level_anchors.to(device)
            anchor_xmin = (grid_x.unsqueeze(1) + level_anchors[:, 0]).flatten()
            anchor_ymin = (grid_y.unsqueeze(1) + level_anchors[:, 1]).flatten()
            anchor_xmax = (grid_x.unsqueeze(1) + level_anchors[:, 2]).flatten()
            anchor_ymax = (grid_y.unsqueeze(1) + level_anchors[:, 3]).flatten()
            
            all_anchors.append(torch.stack((anchor_xmin, anchor_ymin, anchor_xmax, anchor_ymax), dim=1))
        return torch.cat(all_anchors)

class RetinaNet(nn.Module):
    """RetinaNetモデル"""
    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = ResNet18Backbone()
        self.fpn = FeaturePyramidNetwork()
        self.anchor_generator = AnchorGenerator(self.fpn.levels)
        self.class_head = DetectionHead(num_channels_per_anchor=num_classes, num_anchors=self.anchor_generator.num_anchors)
        self.box_head = DetectionHead(num_channels_per_anchor=4, num_anchors=self.anchor_generator.num_anchors)
        self._reset_parameters()

    def _reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        prior = 0.01
        nn.init.zeros_(self.class_head.out_conv.weight)
        nn.init.constant_(self.class_head.out_conv.bias, -torch.log(torch.tensor((1.0 - prior) / prior)))
        nn.init.zeros_(self.box_head.out_conv.weight)
        nn.init.zeros_(self.box_head.out_conv.bias)

    def forward(self, x: torch.Tensor):
        cs = self.backbone(x)
        ps = self.fpn(*cs)
        preds_class = torch.cat(list(map(self.class_head, ps)), dim=1)
        preds_box = torch.cat(list(map(self.box_head, ps)), dim=1)
        feature_sizes = [p.shape[2:] for p in ps]
        anchors = self.anchor_generator.generate(feature_sizes, x.device)
        return preds_class, preds_box, anchors

    def get_device(self):
        return self.backbone.conv1.weight.device

def predict_post_process(preds_class: torch.Tensor, preds_box: torch.Tensor, anchors: torch.Tensor, conf_threshold: float = 0.3, nms_threshold: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    anchors_xywh = convert_to_xywh(anchors)
    preds_box[:, :2] = anchors_xywh[:, :2] + preds_box[:, :2] * anchors_xywh[:, 2:]
    preds_box[:, 2:] = preds_box[:, 2:].exp() * anchors_xywh[:, 2:]
    boxes = convert_to_xyxy(preds_box)
    
    scores, labels = preds_class.sigmoid().max(dim=1)
    keep = scores > conf_threshold
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
    keep_indices = batched_nms(boxes, scores, labels, nms_threshold)
    
    return boxes[keep_indices], scores[keep_indices], labels[keep_indices]