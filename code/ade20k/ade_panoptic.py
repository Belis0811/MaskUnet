import os
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
from panopticapi.utils import id2rgb


BATCH_SIZE = 8
IMG_WIDTH = 128
IMG_HEIGHT = 128

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

ADEK_ROOT = '../../data/ADEK'
IMG_DIR_TRAIN = os.path.join(ADEK_ROOT, 'images', 'training')
ANN_DIR_TRAIN = os.path.join(ADEK_ROOT, 'annotations', 'training')
IMG_DIR_VAL = os.path.join(ADEK_ROOT, 'images', 'validation')
ANN_DIR_VAL = os.path.join(ADEK_ROOT, 'annotations', 'validation')
objectinfo_path = os.path.join(ADEK_ROOT, 'objectInfo150.txt')



def generate_instance_mask(semantic_mask):
    instance_mask = np.zeros_like(semantic_mask, dtype=np.int32)
    unique_classes = np.unique(semantic_mask)
    for class_id in unique_classes:
        if class_id == 0:  # skip background
            continue
        binary_mask = (semantic_mask == class_id).astype(np.uint8)
        num_labels, labels = cv2.connectedComponents(binary_mask)
        # For each connected component (excluding background label 0)
        for label in range(1, num_labels):
            instance_mask[labels == label] = label
    return instance_mask



class ADE20KPanopticDataset(Dataset):
    def __init__(self, objectinfo_txt, img_dir, ann_dir, transforms=None, img_size=(128, 128)):
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.transforms = transforms
        self.img_width, self.img_height = img_size

        # Load category names from objectInfo150.txt
        with open(objectinfo_txt, 'r') as f:
            categories = [line.strip() for line in f if line.strip()]
        self.categories = categories
        # In ADE20K the mask pixel values already denote the category id.
        # Build a mapping (here, identity mapping).
        self.cat2label = {i: i for i in range(len(categories))}

        # List all image files (adjust extension if needed)
        self.img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg') or f.endswith('.png')])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_filename = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_filename)
        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            raise FileNotFoundError(f"Could not find image: {img_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Assume annotation file has the same basename with .png extension
        ann_filename = os.path.splitext(img_filename)[0] + '.png'
        ann_path = os.path.join(self.ann_dir, ann_filename)
        semantic_mask = cv2.imread(ann_path, cv2.IMREAD_GRAYSCALE)
        if semantic_mask is None:
            raise FileNotFoundError(f"Could not find annotation: {ann_path}")

        # Resize image and mask
        image_rgb = cv2.resize(image_rgb, (self.img_width, self.img_height), interpolation=cv2.INTER_LINEAR)
        semantic_mask = cv2.resize(semantic_mask, (self.img_width, self.img_height), interpolation=cv2.INTER_NEAREST)

        # Generate instance mask using connected components
        instance_mask = generate_instance_mask(semantic_mask)

        if self.transforms:
            image_rgb = self.transforms(image_rgb)

        semantic_mask = torch.from_numpy(semantic_mask).long()
        instance_mask = torch.from_numpy(instance_mask).long()

        return image_rgb, semantic_mask, instance_mask


train_dataset = ADE20KPanopticDataset(
    objectinfo_txt=objectinfo_path,
    img_dir=IMG_DIR_TRAIN,
    ann_dir=ANN_DIR_TRAIN,
    transforms=ToTensor(),
    img_size=(IMG_HEIGHT, IMG_WIDTH)
)
val_dataset = ADE20KPanopticDataset(
    objectinfo_txt=objectinfo_path,
    img_dir=IMG_DIR_VAL,
    ann_dir=ANN_DIR_VAL,
    transforms=ToTensor(),
    img_size=(IMG_HEIGHT, IMG_WIDTH)
)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)


def visualize_random_sample(dataset):
    idx = random.randint(0, len(dataset) - 1)
    image, sem_mask, inst_mask = dataset[idx]
    if isinstance(image, torch.Tensor):
        image_np = image.permute(1, 2, 0).cpu().numpy()
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
    else:
        image_np = image

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(image_np)
    plt.title("Image")

    plt.subplot(1, 3, 2)
    plt.imshow(sem_mask.cpu().numpy() if isinstance(sem_mask, torch.Tensor) else sem_mask)
    plt.title("Semantic Mask")

    plt.subplot(1, 3, 3)
    plt.imshow(inst_mask.cpu().numpy() if isinstance(inst_mask, torch.Tensor) else inst_mask)
    plt.title("Instance Mask")
    plt.show()


visualize_random_sample(train_dataset)

def mask_to_rle(binary_mask):
    import pycocotools.mask as maskUtils
    rle = maskUtils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    if isinstance(rle["counts"], bytes):
        rle["counts"] = rle["counts"].decode("utf-8")
    return rle

def get_instances_from_mask(mask, prob_map=None):
    instances = []
    unique_labels = np.unique(mask)
    for label in unique_labels:
        if label == 0:
            continue
        binary_mask = (mask == label).astype(np.uint8)
        num_components, comp_map = cv2.connectedComponents(binary_mask)
        for comp_id in range(1, num_components):
            inst_mask = (comp_map == comp_id).astype(np.uint8)
            if inst_mask.sum() == 0:
                continue
            ys, xs = np.where(inst_mask)
            x_min, x_max = int(xs.min()), int(xs.max())
            y_min, y_max = int(ys.min()), int(ys.max())
            bbox = [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)]
            score = 1.0
            if prob_map is not None:
                score = float(prob_map[label, inst_mask==1].mean())
            instances.append({
                "bbox": bbox,
                "category_id": int(label),
                "segmentation": mask_to_rle(inst_mask),
                "score": score
            })
    return instances

def compute_iou_for_image(pred_mask, gt_mask, num_classes, smooth=1e-6):
    iou_list = []
    for class_id in range(num_classes):
        pred = (pred_mask == class_id).astype(np.uint8)
        gt = (gt_mask == class_id).astype(np.uint8)
        intersection = np.sum((pred==1) & (gt==1))
        union = np.sum((pred==1) | (gt==1))
        if union == 0:
            continue
        iou_list.append((intersection + smooth) / (union + smooth))
    return np.mean(iou_list) if iou_list else 1.0


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
            processed_mask = torch.where(binary_mask > 0.5, torch.tensor(0.0, device=x.device),
                                         torch.tensor(-float('inf'), device=x.device))
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
        torch.save(model.state_dict(), "checkpoint_ade_pan.pth")


class InstanceContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
        self.loss_fn = nn.TripletMarginLoss(margin=self.margin)

    def forward(self, sem_logits, instance_mask):
        unique_instances = torch.unique(instance_mask)
        loss = 0.0
        count = 0
        for inst in unique_instances:
            if inst == 0:
                continue
            inst_pixels = (instance_mask == inst).nonzero(as_tuple=True)
            if len(inst_pixels[0]) < 2:
                continue
            anchor = sem_logits[:, :, inst_pixels[0][0], inst_pixels[1][0]]
            positive = sem_logits[:, :, inst_pixels[0][1], inst_pixels[1][1]]
            negative_pixels = (instance_mask != inst).nonzero(as_tuple=True)
            if len(negative_pixels[0]) == 0:
                continue
            negative_idx = torch.randint(0, len(negative_pixels[0]), (1,))
            negative = sem_logits[:, :, negative_pixels[0][negative_idx], negative_pixels[1][negative_idx]]
            anchor = anchor.view(1, -1)
            positive = positive.view(1, -1)
            negative = negative.view(1, -1)
            loss += self.loss_fn(anchor, positive, negative)
            count += 1
        return loss / count if count > 0 else torch.tensor(0.0, device=sem_logits.device)


c_in = 3
c_out = len(train_dataset.cat2label)
print(f"Total number of classes used: {c_out}")
if torch.cuda.device_count() >= 1:
    device1 = torch.device("cuda:0")
    device2 = torch.device("cuda:1")
else:
    raise RuntimeError("Multiple GPUs are required.")

model = UNet(c_in, c_out)

if os.path.exists("checkpoint_ade_pan.pth"):
    checkpoint = torch.load("checkpoint_ade_pan.pth")
    modified_state_dict = {key.replace("module.", ""): value for key, value in checkpoint.items()}
    model.load_state_dict(modified_state_dict)

import torch.backends.cudnn as cudnn

cudnn.benchmark = True
model = torch.nn.DataParallel(model)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-3)
early_stopping = EarlyStopping(patience=10, verbose=True)
instance_loss_fn = InstanceContrastiveLoss()

num_epochs = 1000
model.to(device1)
best_loss = float("inf")
best_iou = 0.0
log_file = open("training_log_ade_pan.txt", "w")
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    total_iou = 0.0
    for i, (inputs, sem_labels, inst_labels) in enumerate(tqdm(train_loader)):
        inputs = inputs.to(device1)
        sem_labels = sem_labels.to(device1)
        inst_labels = inst_labels.to(device1)
        optimizer.zero_grad()
        sem_logits = model(inputs)
        semantic_loss = criterion(sem_logits, sem_labels)
        inst_loss = instance_loss_fn(sem_logits, inst_labels)
        loss = 0.9 * semantic_loss + 0.1 * inst_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        iou = mean_iou(sem_logits, sem_labels, c_out)
        total_iou += iou.item()
        if i % 500 == 0 and i != 0:
            print(f"Epoch {epoch+1}: Batch[{i}/{len(train_loader)}] Loss: {total_loss / i} IoU: {total_iou / i}")
            if i % 2000 == 0:
                print("Saving model...")
                torch.save(model.state_dict(), "checkpoint_ade_pan.pth")
    avg_loss = total_loss / len(train_loader)
    avg_iou = total_iou / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss} IoU: {avg_iou}")
    best_loss = min(best_loss, avg_loss)
    best_iou = max(best_iou, avg_iou)
    if best_iou - avg_iou > 0.1:
        print("Model Overfit")
        break
    torch.save(model.state_dict(), "epoch_ade_pan.pth")
    log_file.write(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss} IoU: {avg_iou}\n")
    log_file.write(f"Best loss: {best_loss}, Best IoU: {best_iou}\n\n")
    log_file.flush()
    if early_stopping(avg_loss, model):
        print("Early stopping triggered")
        break
print(f"Best loss is {best_loss}, best IoU is {best_iou}")
log_file.close()

model = model.to(device1)
best_val_loss = float("inf")
best_val_iou = 0.0
log_file = open("validating_log_ade_pan.txt", "w")
model.eval()
for epoch in range(num_epochs):
    total_val_loss = 0.0
    total_val_iou = 0.0
    num_batches = len(val_loader)
    with torch.no_grad():
        for inputs, labels, _ in tqdm(val_loader):
            inputs = inputs.to(device1)
            labels = labels.to(device1)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_val_iou += mean_iou(outputs, labels, c_out).item()
            total_val_loss += loss.item()
    avg_val_loss = total_val_loss / num_batches
    avg_val_iou = total_val_iou / num_batches
    print(f"Validation Loss: {avg_val_loss}, Validation IoU: {avg_val_iou}")
    best_val_loss = min(best_val_loss, avg_val_loss)
    best_val_iou = max(best_val_iou, avg_val_iou)
    log_file.write(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_val_loss} IoU: {avg_val_iou}\n")
    log_file.write(f"Best loss: {best_val_loss}, Best IoU: {best_val_iou}\n\n")
    log_file.flush()
print(f"Best Validation loss is {best_val_loss}, best Validation IoU is {best_val_iou}")
log_file.close()

def evaluate_panoptic_metrics(model, data_loader, dataset, device, max_queries=100):
    model.eval()
    all_predictions = []
    all_gt_annotations = []
    image_ids = []
    iou_scores = []
    for batch_idx, (images, gt_sem_masks, _) in enumerate(tqdm(data_loader, desc="Evaluating")):
        images = images.to(device)
        with torch.no_grad():
            outputs = model(images)  # [B, c, H, W]
            probs = torch.softmax(outputs/0.5, dim=1).cpu().numpy()  # [B, c, H, W]
            preds = np.argmax(probs, axis=1)  # [B, H, W]
        for i in range(len(images)):
            gt_mask = gt_sem_masks[i].cpu().numpy()
            pred_mask = preds[i]
            # Compute per-image IoU
            iou = compute_iou_for_image(pred_mask, gt_mask, c_out)
            iou_scores.append(iou)
            image_id = batch_idx * data_loader.batch_size + i
            image_ids.append(image_id)
            pred_instances = get_instances_from_mask(pred_mask, prob_map=probs[i])
            gt_instances = get_instances_from_mask(gt_mask, prob_map=None)
            pred_instances = sorted(pred_instances, key=lambda x: x["score"], reverse=True)[:max_queries]
            for inst in pred_instances:
                inst["image_id"] = image_id
            for inst in gt_instances:
                inst["image_id"] = image_id
            all_predictions.extend(pred_instances)
            all_gt_annotations.extend(gt_instances)
    gt_coco = {
         "images": [{"id": img_id} for img_id in image_ids],
         "annotations": [],
         "categories": [{"id": i, "name": dataset.categories[i]} for i in range(len(dataset.categories))]
    }
    ann_id = 1
    for ann in all_gt_annotations:
         ann["id"] = ann_id
         ann_id += 1
         gt_coco["annotations"].append(ann)
    with open("gt_panoptic.json", "w") as f:
         json.dump(gt_coco, f)
    pred_coco = {
         "images": [{"id": img_id} for img_id in image_ids],
         "annotations": [],
         "categories": [{"id": i, "name": dataset.categories[i]} for i in range(len(dataset.categories))]
    }
    ann_id = 1
    for ann in all_predictions:
         ann["id"] = ann_id
         ann_id += 1
         pred_coco["annotations"].append(ann)
    with open("pred_panoptic.json", "w") as f:
         json.dump(pred_coco, f)
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    coco_gt = COCO("gt_panoptic.json")
    coco_dt = coco_gt.loadRes(pred_coco)
    coco_eval = COCOeval(coco_gt, coco_dt, "segm")
    coco_eval.params.iouThrs = np.linspace(0.5, 0.95, 10)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    ap = coco_eval.stats[0]
    from panopticapi.evaluation import pq_compute
    pq_results = pq_compute(gt_json_file="gt_panoptic.json", pred_json_file="pred_panoptic.json")
    avg_iou = np.mean(iou_scores)
    return ap, pq_results, avg_iou

ap, pq_results, avg_iou = evaluate_panoptic_metrics(model, val_loader, val_dataset, device1, max_queries=100)
print("\nEvaluation Metrics:")
print(f"AP: {ap}")
print(f"PQ: {pq_results['All']} (Things: {pq_results['Things']}, Stuff: {pq_results['Stuff']})")
print(f"Mean IoU: {avg_iou}")

def visualize_predictions(model, dataset, device, idx=0):
    model.eval()
    with torch.no_grad():
        image, mask, _ = dataset[idx]
        image = image.unsqueeze(0).to(device)
        output = model(image)
        print(output.shape)
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
