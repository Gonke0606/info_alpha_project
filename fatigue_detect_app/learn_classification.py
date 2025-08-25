'''
画像分類モデル（ResNet18）の学習
'''
from collections import deque
import copy
from tqdm import tqdm
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
import torchvision.transforms as T

from models_classification import ResNet18

DATA_ROOT = '/Users/wakabayashikengo/fatigue_detection/data_emotion/train' # データが入っているディレクトリのパス
SAVE_PATH = '/Users/wakabayashikengo/fatigue_detection/emotion.pth' # 学習したモデルのパラメータを保存するディレクトリのパス
test_ratio = 0.1 # データのうち検証に使う割合
IMAGE_SIZE = 224

BETAS = (0.9, 0.999) # モーメンタムの更新度合い
WEIGHT_DECAY = 1e-5 # 荷重減衰の度合い

class Config:
    def __init__(self):
        self.val_ratio = 0.2
        self.num_epochs = 50
        self.lr = 1e-3
        self.moving_avg = 20
        self.batch_size = 32
        self.num_workers = 3
        self.lr_drop = 27
        if torch.backends.mps.is_available():
            self.device = 'mps'
        elif torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.num_samples = 200

# データセットの平均値と標準偏差を計算する関数
def get_dataset_statistics(dataset: Dataset):
    data = []
    for i in range(len(dataset)):
        img = dataset[i][0]
        data.append(img)
    data = torch.stack(data)

    channel_mean = data.mean(dim=(0, 2, 3))
    channel_std = data.std(dim=(0, 2, 3))

    return channel_mean, channel_std

def train_eval():
    config = Config()
    
    transforms_for_stats = T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
    ])
    
    full_dataset_for_stats = torchvision.datasets.ImageFolder(
        root=DATA_ROOT,
        transform=transforms_for_stats
    )

    channel_mean, channel_std = get_dataset_statistics(full_dataset_for_stats)
    print(f"データセットの平均値と標準偏差: mean={channel_mean}, std={channel_std}")

    train_transforms = T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        T.ToTensor(),
        T.Normalize(mean=channel_mean, std=channel_std),
    ])

    test_transforms = T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=channel_mean, std=channel_std),
    ])
    
    train_val_dataset = torchvision.datasets.ImageFolder(root=DATA_ROOT, transform=train_transforms)
    test_dataset = torchvision.datasets.ImageFolder(root=DATA_ROOT, transform=test_transforms)

    test_indices, train_val_indices = util.generate_subset(train_val_dataset, test_ratio)
    val_ratio_in_train_val = config.val_ratio / (1 - test_ratio)
    val_indices, train_indices = util.generate_subset_from_indices(train_val_indices, val_ratio_in_train_val)

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = DataLoader(train_val_dataset, batch_size=config.batch_size, num_workers=config.num_workers, sampler=train_sampler)
    val_loader = DataLoader(train_val_dataset, batch_size=config.batch_size, num_workers=config.num_workers, sampler=val_sampler)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, num_workers=config.num_workers, sampler=test_sampler)
    
    loss_func = F.cross_entropy
    val_loss_best = float('inf')
    model_best = None

    num_classes = len(train_val_dataset.classes)
    print(f"{num_classes}個のクラスを分類: {train_val_dataset.classes}")
    
    model = ResNet18(num_classes)
    model.to(config.device)
    
    optimizer = optim.Adam(model.parameters(), lr=config.lr, betas=BETAS, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[config.lr_drop], gamma=0.1)

    # 学習ループ
    for epoch in range(config.num_epochs):
        model.train()
        losses = deque()
        accs = deque()

        with tqdm(train_loader) as pbar:
            pbar.set_description(f'[エポック {epoch + 1}]')
            for x, y in pbar:
                x, y = x.to(config.device), y.to(config.device)
                optimizer.zero_grad()
                y_pred = model(x)
                loss = loss_func(y_pred, y)
                accuracy = (y_pred.argmax(dim=1) == y).float().mean()
                loss.backward()
                optimizer.step()

                losses.append(loss.item())
                accs.append(accuracy.item())
                if len(losses) > config.moving_avg:
                    losses.popleft(), accs.popleft()
                pbar.set_postfix({'loss(avg)': torch.Tensor(losses).mean().item(), 'acc(avg)': torch.Tensor(accs).mean().item()})

        val_loss, val_accuracy = eval.evaluate(val_loader, model, loss_func)
        print(f'検証: loss = {val_loss:.3f}, accuracy = {val_accuracy:.3f}')

        if val_loss < val_loss_best:
            print(f"  -> 検証ロスが改善しました ({val_loss_best:.3f} -> {val_loss:.3f})。モデルを保存します。")
            val_loss_best = val_loss
            model_best = model.copy()
            scheduler.step()

    test_loss, test_accuracy = eval.evaluate(test_loader, model_best, loss_func)
    print(f'テスト: loss = {test_loss:.3f}, accuracy = {test_accuracy:.3f}')

    torch.save(model_best.state_dict(), SAVE_PATH)

def main():
    train_eval()

if __name__ == "__main__":
    main()