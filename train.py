import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset

# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# データセットのロード
dataset = load_dataset('Maysee/tiny-imagenet', split='train')

# データローダーの作成
class TinyImageNetDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x)  # グレースケール画像をRGBに変換
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]['image']
        label = self.dataset[idx]['label']
        image = self.transform(image)
        return image, label

train_dataset = TinyImageNetDataset(dataset)
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)

# gMLPモデルの定義
class gMLPBlock(nn.Module):
    def __init__(self, dim, seq_len, hdim):
        super(gMLPBlock, self).__init__()
        self.fc1 = nn.Linear(dim, hdim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hdim, dim)
        self.norm = nn.LayerNorm(dim)
        self.sgu = SGU(seq_len, dim)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.sgu(x)
        x = x + residual
        return x

class SGU(nn.Module):
    def __init__(self, seq_len, dim):
        super(SGU, self).__init__()
        self.split_dim = dim // 2
        self.conv = nn.Conv1d(self.split_dim, self.split_dim, kernel_size=1)

    def forward(self, x):
        u, v = x.chunk(2, dim=-1)
        v = v.permute(0, 2, 1)  # (B, split_dim, seq_len)
        v = self.conv(v)
        v = v.permute(0, 2, 1)  # (B, seq_len, split_dim)
        return torch.cat((u, v), dim=-1)

class gMLP(nn.Module):
    def __init__(self, num_classes=200):
        super(gMLP, self).__init__()
        self.embedding = nn.Linear(3*16*16, 512)
        self.gmlp_blocks = nn.ModuleList([gMLPBlock(512, 16, 2048) for _ in range(4)])
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = x.unfold(2, 16, 16).unfold(3, 16, 16)
        x = x.contiguous().view(B, 16, -1)
        x = self.embedding(x)
        for block in self.gmlp_blocks:
            x = block(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x

# モデルの初期化
model = gMLP(num_classes=200).to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# トレーニングループ
def train(model, train_loader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}')

train(model, train_loader, criterion, optimizer)
