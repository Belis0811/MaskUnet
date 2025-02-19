import os
import numpy as np
import warnings
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from pycocotools.coco import COCO
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import json
from panopticapi.utils import rgb2id
from pycocotools.cocoeval import COCOeval
from panopticapi.evaluation import pq_compute
import torchvision.transforms as T
import pycocotools.mask as maskUtils

BATCH_SIZE = 4
IMG_WIDTH = 128
IMG_HEIGHT = 128
TRAIN_PATH = './COCO/'
TEST_PATH = './COCO/'
warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42

class COCOSegmentationDataset(Dataset):
    def __init__(self, json_file, image_dir, transforms=None):
        with open(json_file, 'r') as f:
            self.coco = COCO(json_file)
        self.image_ids = list(self.coco.imgs.keys())
        self.image_dir = image_dir
        if transforms is None:
            self.transforms = T.Compose([
                T.ToPILImage(),
                T.Resize((IMG_HEIGHT, IMG_WIDTH)),
                T.ToTensor()
            ])
        else:
            self.transforms = transforms
    def __len__(self):
        return len(self.image_ids)
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.image_dir, image_info['file_name'])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        annotation_ids = self.coco.getAnnIds(imgIds=image_id)
        annotations = self.coco.loadAnns(annotation_ids)
        masks = []
        labels = []
        for ann in annotations:
            mask = self.coco.annToMask(ann)
            masks.append(mask)
            labels.append(ann['category_id'])
        masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_pil = self.transforms(image)
        resized_masks = []
        for m in masks:
            if isinstance(m, torch.Tensor):
                m = m.cpu().numpy()
            m_resized = cv2.resize(m, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_NEAREST)
            resized_masks.append(m_resized)
        resized_masks = np.array(resized_masks, dtype=np.uint8)
        resized_masks = torch.as_tensor(resized_masks)
        labels = torch.as_tensor(labels)
        target = {
            'masks': resized_masks,
            'labels': labels,
            'image_id': torch.tensor([image_id])
        }
        return image_pil, target

train_dataset = COCOSegmentationDataset(json_file='./COCO/annotations/instances_train2017.json', image_dir='./COCO/train2017')
val_dataset = COCOSegmentationDataset(json_file='./COCO/annotations/instances_val2017.json', image_dir='./COCO/val2017')
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, collate_fn=lambda batch: tuple(zip(*batch)))
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, collate_fn=lambda batch: tuple(zip(*batch)))
import random
import matplotlib.pyplot as plt
import numpy as np

def visualize_random_sample(dataset):
    idx = random.randint(0, len(dataset) - 1)
    image, mask = dataset[idx]
    if isinstance(image, torch.Tensor):
        image_np = image.permute(1, 2, 0).cpu().numpy()
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
    else:
        image_np = image
    combined_mask = torch.sum(mask['masks'], dim=0)
    combined_mask_np = combined_mask.cpu().numpy().astype(np.uint8)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(image_np)
    plt.title("Image")
    plt.subplot(1, 2, 2)
    plt.imshow(combined_mask_np)
    plt.title("Mask")
    plt.show()

visualize_random_sample(train_dataset)

def mean_iou(y_pred, y_true, num_classes, smooth=1e-6):
    y_pred = torch.softmax(y_pred/0.5, dim=1)
    y_pred = torch.argmax(y_pred, dim=1)
    iou_list = []
    for class_id in range(num_classes):
        intersection = torch.sum((y_pred == class_id) & (y_true == class_id))
        union = torch.sum((y_pred == class_id) | (y_true == class_id))
        if union == 0:
            continue
        iou = (intersection.float() + smooth) / (union.float() + smooth)
        iou_list.append(iou)
    return torch.mean(torch.stack(iou_list))

import torch
import torch.nn as nn
import torch.nn.functional as F

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
            raise ValueError("Input channel size does not match initialized channel size.")
        x = x.view(batch_size, channels, height * width).permute(0, 2, 1)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores / (self.channels ** 0.5)
        if self.mask is None or self.mask.size(-1) != height * width:
            binary_mask = torch.randint(0, 2, (batch_size, height, width), device=x.device)
            binary_mask = binary_mask.view(batch_size, -1)
            processed_mask = torch.where(binary_mask > 0.5, torch.tensor(0.0, device=x.device), torch.tensor(-float('inf'), device=x.device))
            self.mask = processed_mask.unsqueeze(1).expand(-1, height * width, -1)
        scores = scores + self.mask
        attention_weights = F.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)
        attention_output = attention_output + x
        attention_output = self.norm(attention_output)
        return attention_output.view(batch_size, channels, height, width)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super(ConvBlock, self).__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.conv_block(x))
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
            nn.Linear(emb_dim, out_channels),
        )
    def forward(self, x):
        x = self.maxpool_conv(x)
        return x

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super(UpSample, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.out_channels = out_channels
        self.conv = nn.Sequential(
            ConvBlock(in_channels, in_channels, residual=True),
            ConvBlock(in_channels, out_channels, in_channels // 2),
            nn.BatchNorm2d(out_channels)
        )
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels),
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
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.save_checkpoint(model)
            self.counter = 0
        return self.early_stop
    def save_checkpoint(self, model):
        if self.verbose:
            print(f'Saving model...')
        torch.save(model.state_dict(), 'checkpoint_instance.pth')

import torch.backends.cudnn as cudnn
from torch_lr_finder import LRFinder
c_in = 3
c_out = len(train_dataset.coco.cats)+1
print(f'total number of classes used: {c_out}')
if torch.cuda.device_count() > 1:
    device1 = torch.device("cuda:0")
    device2 = torch.device("cuda:1")
else:
    raise RuntimeError("Multiple GPUs are required.")
model = UNet(c_in, c_out)
model_dict = model.state_dict()

checkpoint = torch.load('checkpoint_instance.pth')
modified_state_dict = {key.replace('module.', ''): value for key, value in checkpoint.items()}
#pretrained_dict = {k: v for k, v in modified_state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
model.load_state_dict(modified_state_dict)
#model_dict.update(pretrained_dict)
#model.load_state_dict(model_dict)

cudnn.benchmark = True
model = torch.nn.DataParallel(model)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
early_stopping = EarlyStopping(patience=10, verbose=True)
num_epochs = 1000
model.to(device1)
best_loss = float("inf")
best_ap = 0.0
log_file = open("train_ap_results.txt", "w")
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    detections = []
    ap_values = 0.0
    for i,(images, targets) in enumerate(tqdm(train_loader)):
        images = torch.stack([img.to(device1) for img in images])
        original_targets = targets
        masks_batch = []
        for t in targets:
            if t['masks'].nelement() > 0:  
                combined_mask = torch.sum(t['masks'], dim=0)
                masks_batch.append(combined_mask)
            else:
                masks_batch.append(torch.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=torch.uint8)) 

        masks_batch = torch.stack(masks_batch).to(device1)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if i % 5000 == 0 and i != 0:
            for j, output in enumerate(outputs):
                image_id = original_targets[j]['image_id'].item()
                # converting to binary for RLE
                score = torch.sigmoid(output).max().item()
                pred_mask = (output.detach().cpu().numpy() > 0.5).astype(np.uint8)
                rle = maskUtils.encode(np.asfortranarray(pred_mask[0]))
                rle['counts'] = rle['counts'].decode('ascii')
                detections.append({
                    'image_id': int(image_id),
                    'category_id': 1,
                    'segmentation': rle,
                    'score': score
                })
            with open('train_detections.json', 'w') as f:
                json.dump(detections, f)
            coco_pred_train = train_dataset.coco.loadRes('train_detections.json')
            coco_eval_train = COCOeval(train_dataset.coco, coco_pred_train, 'segm')
            coco_eval_train.evaluate()
            coco_eval_train.accumulate()
            coco_eval_train.summarize()
            ap_values = coco_eval_train.stats[:6]
            
            print(f"Epoch {epoch+1}: Batch[{i}/{len(train_loader)}] Loss: {total_loss / i}")
            if i % 2000 == 0:
                print(f'Saving model...')
                torch.save(model.state_dict(), 'checkpoint_instance.pth')
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss}")
    best_loss = min(best_loss, avg_loss)
    #best_ap = max(best_ap, ap_values)
    torch.save(model.state_dict(), 'epoch_instance.pth')
    log_file.write(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss} AP values: {ap_values}\n")
    log_file.write(f'Best loss: {best_loss}\n\n')
    log_file.flush()
    if early_stopping(avg_loss, model):
        print('Early stopping triggered')
        break
print(f'Best loss is {best_loss}')
log_file.close()
model = model.to(device1)
model.eval()
coco_true = val_dataset.coco
detections = []
with torch.no_grad():
    for images, targets in tqdm(val_loader):
        images = torch.stack([img.to(device1) for img in images])
        outputs = model(images)
        for i, output in enumerate(outputs):
            image_id = targets[i]['image_id'].item()
            pred_mask = (output.detach().cpu().numpy() > 0.5).astype(np.uint8)
            rle = maskUtils.encode(np.asfortranarray(pred_mask[0]))
            rle['counts'] = rle['counts'].decode('ascii')
            detections.append({
                'image_id': int(image_id),
                'category_id': 1,
                'segmentation': rle,
                'score': 0.95
            })
with open('detections.json', 'w') as f:
    json.dump(detections, f)
coco_pred = coco_true.loadRes('detections.json')
coco_eval = COCOeval(coco_true, coco_pred, 'segm')
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()
pq_results = pq_compute(gt_json_file='val_annotations.json', pred_json_file='detections.json')
print("Panoptic Quality:", pq_results)
