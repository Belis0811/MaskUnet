import os
import glob
import numpy as np
import warnings
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import json

# -------------------------------
# Parameters and dataset paths
# -------------------------------
BATCH_SIZE = 1
IMG_WIDTH = 128
IMG_HEIGHT = 128

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42
random.seed(seed)

CITYSCAPES_ROOT = '../../data/Cityscapes'
IMG_DIR_TRAIN = os.path.join(CITYSCAPES_ROOT, 'leftImg8bit_trainvaltest','leftImg8bit', 'train')
ANN_DIR_TRAIN = os.path.join(CITYSCAPES_ROOT, 'gtFine_trainvaltest','gtFine', 'train')
IMG_DIR_VAL = os.path.join(CITYSCAPES_ROOT, 'leftImg8bit_trainvaltest','leftImg8bit', 'val')
ANN_DIR_VAL = os.path.join(CITYSCAPES_ROOT, 'gtFine_trainvaltest','gtFine', 'val')

CITYSCAPES_CLASSES = [
    "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
    "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
    "truck", "bus", "train", "motorcycle", "bicycle"
]

# -------------------------------
# Cityscapes Instance Segmentation Dataset
# -------------------------------
class CityscapesSegmentationDataset(Dataset):
    def __init__(self, img_dir, ann_dir, transforms=None, img_size=(128, 128)):
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.transforms = transforms
        self.img_width, self.img_height = img_size

        # List all image files recursively from the image directory.
        self.img_files = sorted(glob.glob(os.path.join(img_dir, '*', '*.png')))
        
        # Store categories and build a mapping.
        self.categories = CITYSCAPES_CLASSES
        self.cat2label = {i: i for i in range(len(self.categories))}

    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            raise FileNotFoundError(f"Could not find {img_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # Compute relative path from the image directory.
        rel_path = os.path.relpath(img_path, self.img_dir)  # e.g., "ulm/ulm_000003_000019_leftImg8bit.png"
        city = os.path.dirname(rel_path)  # e.g., "ulm"
        img_basename = os.path.basename(rel_path)  # e.g., "ulm_000003_000019_leftImg8bit.png"
        
        # Remove "leftImg8bit" from the base name.
        base = os.path.splitext(img_basename)[0].replace("leftImg8bit", "")
        base = base.strip('_')  # remove any leading/trailing underscores if needed
        ann_filename = base + "_gtFine_labelIds.png"
        ann_path = os.path.join(self.ann_dir, city, ann_filename)
        
        if not os.path.exists(ann_path):
            raise FileNotFoundError(f"Could not find {ann_path}")
        
        semantic_mask = cv2.imread(ann_path, cv2.IMREAD_GRAYSCALE)
        if semantic_mask is None:
            raise FileNotFoundError(f"Could not load annotation from {ann_path}")
        
        semantic_mask = np.where((semantic_mask >= 0) & (semantic_mask < 19), semantic_mask, 255)
        
        # Resize image and mask.
        image_rgb = cv2.resize(image_rgb, (self.img_width, self.img_height), interpolation=cv2.INTER_LINEAR)
        semantic_mask = cv2.resize(semantic_mask, (self.img_width, self.img_height), interpolation=cv2.INTER_NEAREST)
        
        if self.transforms:
            image_rgb = self.transforms(image_rgb)
        else:
            image_rgb = ToTensor()(image_rgb)
        
        semantic_mask = torch.from_numpy(semantic_mask).long()
        return image_rgb, semantic_mask


train_dataset = CityscapesSegmentationDataset(
    img_dir=IMG_DIR_TRAIN,
    ann_dir=ANN_DIR_TRAIN,
    transforms=ToTensor(),
    img_size=(IMG_HEIGHT, IMG_WIDTH)
)

val_dataset = CityscapesSegmentationDataset(
    img_dir=IMG_DIR_VAL,
    ann_dir=ANN_DIR_VAL,
    transforms=ToTensor(),
    img_size=(IMG_HEIGHT, IMG_WIDTH)
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

# -------------------------------
# Visualization Function
# -------------------------------
def visualize_random_sample(dataset):
    idx = random.randint(0, len(dataset) - 1)
    image, mask = dataset[idx]
    if isinstance(image, torch.Tensor):
        image_np = image.permute(1, 2, 0).cpu().numpy()
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
    else:
        image_np = image
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(image_np)
    plt.title("Image")
    plt.subplot(1, 2, 2)
    plt.imshow(mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask)
    plt.title("Mask")
    plt.show()

visualize_random_sample(train_dataset)

# -------------------------------
# UNet Model with Self-Attention Blocks (similar to ADE20K code)
# -------------------------------
class Mask2FormerAttention(nn.Module):
    def __init__(self, channels, size):
        super(Mask2FormerAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.query = nn.Linear(channels, channels)
        self.key = nn.Linear(channels, channels)
        self.value = nn.Linear(channels, channels)
        self.mask = None  
        self.norm = nn.LayerNorm([channels])

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        if channels != self.channels:
            raise ValueError("Input channel size does not match.")
        x = x.view(batch_size, channels, height * width).permute(0, 2, 1)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores / (self.channels ** 0.5)
        if self.mask is None or self.mask.size(-1) != height * width:
            binary_mask = torch.randint(0, 2, (batch_size, height, width), device=x.device)
            binary_mask = binary_mask.view(batch_size, -1)
            processed_mask = torch.where(binary_mask > 0.5, torch.tensor(0.0, device=x.device), 
                                           torch.tensor(-float('inf'), device=x.device))
            self.mask = processed_mask.unsqueeze(1).expand(-1, height * width, -1)
        scores = scores + self.mask
        attention_weights = torch.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)
        attention_output = attention_output + x
        attention_output = self.norm(attention_output)
        return attention_output.view(batch_size, channels, height, width)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super(ConvBlock, self).__init__()
        self.residual = residual
        if mid_channels is None:
            mid_channels = out_channels
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        if self.residual:
            return nn.functional.gelu(x + self.conv_block(x))
        else:
            return self.conv_block(x)

class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super(DownSample, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, in_channels, residual=True),
            ConvBlock(in_channels, out_channels),
            nn.BatchNorm2d(out_channels)
        )
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels)
        )

    def forward(self, x):
        x = self.maxpool_conv(x)
        return x

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super(UpSample, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            ConvBlock(in_channels, in_channels, residual=True),
            ConvBlock(in_channels, out_channels, in_channels // 2),
            nn.BatchNorm2d(out_channels)
        )
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels)
        )

    def forward(self, x, skip_x):
        x = self.upsample(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        return x

class UNet(nn.Module):
    def __init__(self, c_in=3, c_out=3):
        super(UNet, self).__init__()
        self.initial_conv = ConvBlock(c_in, 64)
        self.downsample1 = DownSample(64, 128)
        self.self_attention1 = Mask2FormerAttention(128, 128)
        self.downsample2 = DownSample(128, 256)
        self.self_attention2 = Mask2FormerAttention(256, 256)
        self.downsample3 = DownSample(256, 256)
        self.self_attention3 = Mask2FormerAttention(256, 256)
        self.bottom1 = ConvBlock(256, 512)
        self.bottom2 = ConvBlock(512, 512)
        self.bottom3 = ConvBlock(512, 256)
        self.dropout = nn.Dropout(0.3)
        self.upsample1 = UpSample(512, 128)
        self.self_attention4 = Mask2FormerAttention(128, 128)
        self.upsample2 = UpSample(256, 64)
        self.self_attention5 = Mask2FormerAttention(64, 64)
        self.upsample3 = UpSample(128, 64)
        self.self_attention6 = Mask2FormerAttention(64, 64)
        self.norm = nn.LayerNorm([64, 128, 128])
        self.final_layer = nn.Sequential(
            nn.Conv2d(64, c_out, kernel_size=1),
            nn.BatchNorm2d(c_out),
            nn.ReLU()
        )

    def forward(self, x):
        x1 = self.initial_conv(x)
        x2 = self.downsample1(x1)
        x2 = self.self_attention1(x2)
        x3 = self.downsample2(x2)
        x3 = self.self_attention2(x3)
        x4 = self.downsample3(x3)
        x4 = self.self_attention3(x4)
        x4 = self.bottom1(x4)
        x4 = self.bottom2(x4)
        x4 = self.bottom3(x4)
        x = self.upsample1(x4, x3)
        x = self.dropout(x)
        x = self.self_attention4(x)
        x = self.upsample2(x, x2)
        x = self.dropout(x)
        x = self.self_attention5(x)
        x = self.upsample3(x, x1)
        x = self.self_attention6(x)
        x = self.norm(x)
        output = self.final_layer(x)
        return output

# -------------------------------
# Early Stopping
# -------------------------------
class EarlyStopping:
    def __init__(self, patience=3, verbose=True):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_score is None:
            self.best_score = val_loss
            self.save_checkpoint(model)
        elif val_loss > self.best_score:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.save_checkpoint(model)
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        if self.verbose:
            print("Saving model...")
        torch.save(model.state_dict(), "checkpoint_cityscapes.pth")

# -------------------------------
# Setup model, loss, optimizer
# -------------------------------
c_in = 3
c_out = len(train_dataset.cat2label)
print(f"Total number of classes used: {c_out}")

if torch.cuda.device_count() >= 1:
    device1 = torch.device("cuda:0")
    device2 = torch.device("cuda:1")
else:
    raise RuntimeError("Multiple GPUs are required.")

model = UNet(c_in, c_out)

model_dict = model.state_dict()

if os.path.exists("checkpoint_city_pan.pth"):
    checkpoint = torch.load('checkpoint_pan.pth')
    modified_state_dict = {key.replace('module.', ''): value for key, value in checkpoint.items()}
    filtered_state_dict = {k: v for k, v in modified_state_dict.items() if not k.startswith('final_layer.')}

    model.load_state_dict(filtered_state_dict, strict=False)

model = torch.nn.DataParallel(model)
criterion = nn.CrossEntropyLoss(ignore_index=255)
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
early_stopping = EarlyStopping(patience=10, verbose=True)

import torch.backends.cudnn as cudnn
cudnn.benchmark = True
model.to(device1)

# -------------------------------
# Training Loop
# -------------------------------
num_epochs = 1000
best_loss = float("inf")
log_file = open("training_log_cityscapes.txt", "w")
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    total_iou = 0.0
    for i, (inputs, labels) in enumerate(tqdm(train_loader)):
        inputs, labels = inputs.to(device1), labels.to(device1)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        # Compute mean IoU on the batch (using softmax and argmax)
        y_pred = torch.softmax(outputs / 0.5, dim=1).argmax(dim=1)
        iou = 0.0
        count = 0
        for class_id in range(c_out):
            intersection = torch.sum((y_pred == class_id) & (labels == class_id))
            union = torch.sum((y_pred == class_id) | (labels == class_id))
            if union == 0:
                continue
            iou += (intersection.float() + 1e-6) / (union.float() + 1e-6)
            count += 1
        if count > 0:
            total_iou += (iou / count).item()
        if i % 500 == 0 and i != 0:
            print(f"Epoch {epoch+1}: Batch [{i}/{len(train_loader)}] Loss: {total_loss / i:.4f} IoU: {total_iou / i:.4f}")
    avg_loss = total_loss / len(train_loader)
    avg_iou = total_iou / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f} IoU: {avg_iou:.4f}")
    best_loss = min(best_loss, avg_loss)
    torch.save(model.state_dict(), "epoch_cityscapes.pth")
    log_file.write(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f} IoU: {avg_iou:.4f}\n")
    log_file.write(f"Best loss: {best_loss:.4f}\n\n")
    log_file.flush()
    if early_stopping(avg_loss, model):
        print("Early stopping triggered")
        break
print(f"Best loss is {best_loss:.4f}")
log_file.close()

# -------------------------------
# Validation Loop
# -------------------------------
model.eval()
num_batches = len(val_loader)
total_val_loss = 0.0
total_val_iou = 0.0
with torch.no_grad():
    for inputs, labels in tqdm(val_loader):
        inputs, labels = inputs.to(device1), labels.to(device1)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        total_val_loss += loss.item()
        y_pred = torch.softmax(outputs / 0.5, dim=1).argmax(dim=1)
        iou = 0.0
        count = 0
        for class_id in range(c_out):
            intersection = torch.sum((y_pred == class_id) & (labels == class_id))
            union = torch.sum((y_pred == class_id) | (labels == class_id))
            if union == 0:
                continue
            iou += (intersection.float() + 1e-6) / (union.float() + 1e-6)
            count += 1
        if count > 0:
            total_val_iou += (iou / count).item()
avg_val_loss = total_val_loss / num_batches
avg_val_iou = total_val_iou / num_batches
print(f"Validation Loss: {avg_val_loss:.4f}, Validation IoU: {avg_val_iou:.4f}")

# -------------------------------
# Visualization of Predictions
# -------------------------------
def visualize_predictions(model, dataset, device, idx=0):
    model.eval()
    with torch.no_grad():
        image, mask = dataset[idx]
        image_input = image.unsqueeze(0).to(device)
        output = model(image_input)
        pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 3, 1)
        plt.imshow(image.squeeze(0).permute(1, 2, 0).cpu().numpy())
        plt.title("Original Image")
        plt.subplot(1, 3, 2)
        plt.imshow(mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask)
        plt.title("Ground Truth")
        plt.subplot(1, 3, 3)
        plt.imshow(pred_mask)
        plt.title("Predicted Mask")
        plt.show()

test_idx = random.randint(0, len(val_dataset) - 1)
visualize_predictions(model, val_dataset, device1, idx=test_idx)
