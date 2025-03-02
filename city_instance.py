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
import cv2
import json
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import pycocotools.mask as maskUtils

# -------------------------------
# Parameters and dataset paths
# -------------------------------
BATCH_SIZE = 11
IMG_WIDTH = 128
IMG_HEIGHT = 128
LAMBDA_IE = 0.5

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

CITYSCAPES_ROOT = './Cityscapes'
IMG_DIR_TRAIN = os.path.join(CITYSCAPES_ROOT, 'leftImg8bit', 'train')
ANN_DIR_TRAIN = os.path.join(CITYSCAPES_ROOT, 'gtFine', 'train')
IMG_DIR_VAL = os.path.join(CITYSCAPES_ROOT, 'leftImg8bit', 'val')
ANN_DIR_VAL = os.path.join(CITYSCAPES_ROOT, 'gtFine', 'val')

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
        
        # Derive the semantic annotation path.
        ann_path = img_path.replace('leftImg8bit', 'gtFine')
        ann_path = ann_path.replace('leftImg8bit.png', 'gtFine_labelTrainIds.png')
        if not os.path.exists(ann_path):
            raise FileNotFoundError(f"Could not find {ann_path}")
        
        semantic_mask = cv2.imread(ann_path, cv2.IMREAD_GRAYSCALE)
        if semantic_mask is None:
            raise FileNotFoundError(f"Could not load annotation from {ann_path}")
        
        # Derive the instance annotation path.
        instance_ann_path = ann_path.replace('labelTrainIds', 'instanceIds')
        instance_mask = cv2.imread(instance_ann_path, cv2.IMREAD_GRAYSCALE)
        if instance_mask is None:
            raise FileNotFoundError(f"Could not load instance annotation from {instance_ann_path}")
        
        # Resize image and masks.
        image_rgb = cv2.resize(image_rgb, (self.img_width, self.img_height), interpolation=cv2.INTER_LINEAR)
        semantic_mask = cv2.resize(semantic_mask, (self.img_width, self.img_height), interpolation=cv2.INTER_NEAREST)
        instance_mask = cv2.resize(instance_mask, (self.img_width, self.img_height), interpolation=cv2.INTER_NEAREST)
        
        if self.transforms:
            image_rgb = self.transforms(image_rgb)
        else:
            image_rgb = ToTensor()(image_rgb)
        
        semantic_mask = torch.from_numpy(semantic_mask).long()
        instance_mask = torch.from_numpy(instance_mask).long()
        return image_rgb, semantic_mask, instance_mask

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
# Model definition (Modified UNet with Boundary and Embedding Heads)
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
    def __init__(self, c_in=3, c_out=3, embed_dim=16):
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
        self.boundary_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1)
        )
        # Extra branch for per-pixel embeddings (for IE loss)
        self.embedding_head = nn.Sequential(
            nn.Conv2d(64, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
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
        # Compute per-pixel embeddings from the shared feature map.
        embeddings = self.embedding_head(x)
        semantic_out = self.final_layer(x)
        boundary_map = self.boundary_head(semantic_out)
        return semantic_out, boundary_map, embeddings

# -------------------------------
# Instance Contrastive (IE) Loss (Instance Embedding Loss)
# -------------------------------
def instance_contrastive_loss(embeddings, instance_labels, temperature=0.1):
    """
    embeddings: Tensor of shape [B, D, H, W]
    instance_labels: Tensor of shape [B, H, W] with instance IDs (0 is background)
    """
    B, D, H, W = embeddings.shape
    total_loss = 0.0
    count = 0
    for b in range(B):
        emb = embeddings[b].view(D, -1).T  # shape: [N, D]
        labels = instance_labels[b].view(-1)  # shape: [N]
        N = emb.shape[0]
        # Compute similarity matrix
        similarity = torch.matmul(emb, emb.T) / temperature  # [N, N]
        exp_sim = torch.exp(similarity)
        # Build positive mask: same instance (non-background)
        positive_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        positive_mask = positive_mask * (labels.unsqueeze(0) != 0).float()
        # Exclude self-similarity (diagonal)
        diag = torch.eye(N, device=emb.device)
        positive_mask = positive_mask - diag
        denom = exp_sim.sum(dim=1) - torch.exp(torch.diag(similarity))
        pos_sim = (exp_sim * positive_mask).sum(dim=1)
        loss_pixel = -torch.log((pos_sim + 1e-6) / (denom + 1e-6))
        valid = (positive_mask.sum(dim=1) > 0).float()
        if valid.sum() > 0:
            loss_b = (loss_pixel * valid).sum() / (valid.sum() + 1e-6)
            total_loss += loss_b
            count += 1
    return total_loss / (count + 1e-6)

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
            print('Saving model...')
        torch.save(model.state_dict(), 'checkpoint_cityscapes_inst.pth')

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

# -------------------------------
# Set up model, loss, optimizer
# -------------------------------
c_in = 3
c_out = len(train_dataset.cat2label)
print(f'Total number of classes used: {c_out}')
if torch.cuda.device_count() > 1:
    device1 = torch.device("cuda:0")
    device2 = torch.device("cuda:1")
else:
    raise RuntimeError("Multiple GPUs are required.")

model = UNet(c_in, c_out, embed_dim=16)
if os.path.exists('checkpoint_cityscapes_inst.pth'):
    checkpoint = torch.load('checkpoint_cityscapes_inst.pth')
    modified_state_dict = {key.replace('module.', ''): value for key, value in checkpoint.items()}
    model.load_state_dict(modified_state_dict)

model = torch.nn.DataParallel(model)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
early_stopping = EarlyStopping(patience=10, verbose=True)

# -------------------------------
# Training Loop
# -------------------------------
num_epochs = 1000
model.to(device1)
best_loss = float("inf")
log_file = open("training_log_cityscapes_inst.txt", "w")
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for i, (inputs, sem_labels, inst_labels) in enumerate(tqdm(train_loader)):
        inputs = inputs.to(device1)
        sem_labels = sem_labels.to(device1)
        inst_labels = inst_labels.to(device1)
        optimizer.zero_grad()
        sem_out, boundary_map, embeddings = model(inputs)
        seg_loss = criterion(sem_out, sem_labels)
        ie_loss = instance_contrastive_loss(embeddings, inst_labels)
        loss = seg_loss + LAMBDA_IE * ie_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        if i % 500 == 0 and i != 0:
            print(f"Epoch {epoch+1}: Batch [{i}/{len(train_loader)}] Loss: {total_loss / i}")
            if i % 200 == 0:
                print('Saving model...')
                torch.save(model.state_dict(), 'checkpoint_cityscapes_inst.pth')
                
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss}")
    best_loss = min(best_loss, avg_loss)
    torch.save(model.state_dict(), 'epoch_cityscapes_inst.pth')
    log_file.write(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss}\n")
    log_file.write(f"Best loss: {best_loss}\n\n")
    log_file.flush()
    if early_stopping(avg_loss, model):
        print('Early stopping triggered')
        break
print(f'Best loss is {best_loss}')
log_file.close()

# -------------------------------
# Helper Functions for Evaluation
# -------------------------------
def mask_to_rle(binary_mask):
    rle = maskUtils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    if isinstance(rle["counts"], bytes):
        rle["counts"] = rle["counts"].decode("utf-8")
    return rle

def get_instances_from_embeddings(semantic_pred, embeddings, eps=0.5, min_samples=5):
    """
    Clusters pixel embeddings (for each semantic class) using DBSCAN.
    semantic_pred: numpy array [H, W] of predicted semantic labels.
    embeddings: numpy array [H, W, D] of per-pixel embeddings.
    Returns an instance mask with unique IDs.
    """
    instance_mask = np.zeros_like(semantic_pred, dtype=np.int32)
    instance_id = 1
    unique_labels = np.unique(semantic_pred)
    for label in unique_labels:
        if label == 0:
            continue
        indices = np.where(semantic_pred == label)
        if len(indices[0]) == 0:
            continue
        pixel_embeddings = embeddings[indices[0], indices[1], :]  # shape: (N, D)
        if len(pixel_embeddings) < min_samples:
            clusters = np.array([-1]*len(pixel_embeddings))
        else:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            clusters = dbscan.fit_predict(pixel_embeddings)
        for cl in np.unique(clusters):
            if cl == -1:
                continue
            mask_cluster = np.zeros_like(semantic_pred, dtype=np.uint8)
            cluster_indices = (clusters == cl)
            mask_cluster[indices[0][cluster_indices], indices[1][cluster_indices]] = 1
            instance_mask[mask_cluster == 1] = instance_id
            instance_id += 1
    return instance_mask

def get_instance_annotations(instance_mask, sem_mask):
    """
    Converts an instance mask into COCO-style annotations.
    instance_mask: numpy array [H, W] with unique instance IDs.
    sem_mask: numpy array [H, W] with semantic labels.
    """
    instances = []
    unique_ids = np.unique(instance_mask)
    for inst_id in unique_ids:
        if inst_id == 0:
            continue
        binary_mask = (instance_mask == inst_id).astype(np.uint8)
        ys, xs = np.where(binary_mask)
        if len(xs) == 0:
            continue
        bbox = [float(xs.min()), float(ys.min()), float(xs.max() - xs.min()), float(ys.max() - ys.min())]
        # Approximate category from semantic mask using median value
        category_id = int(np.median(sem_mask[binary_mask == 1]))
        instances.append({
            "bbox": bbox,
            "category_id": category_id,
            "segmentation": mask_to_rle(binary_mask),
            "score": 1.0
        })
    return instances

def evaluate_instances(model, data_loader, device, max_queries):
    model.eval()
    all_predictions = []
    all_gt = []
    image_ids = []
    global_img_id = 0
    for images, gt_sem_masks, gt_inst_masks in tqdm(data_loader, desc="Evaluating"):
        images = images.to(device)
        with torch.no_grad():
            sem_out, _, embeddings = model(images)  # sem_out: [B, c, H, W]; embeddings: [B, D, H, W]
            probs = torch.softmax(sem_out / 0.5, dim=1).cpu().numpy()
            sem_preds = np.argmax(probs, axis=1)  # [B, H, W]
            embeddings = embeddings.cpu()  # [B, D, H, W]
        batch_size = images.shape[0]
        for i in range(batch_size):
            image_id = global_img_id
            global_img_id += 1
            sem_pred = sem_preds[i]
            emb = embeddings[i].permute(1, 2, 0).numpy()  # [H, W, D]
            # Cluster embeddings to obtain instance segmentation mask.
            inst_mask = get_instances_from_embeddings(sem_pred, emb, eps=0.5, min_samples=5)
            # Get annotations from predicted instance mask.
            pred_instances = get_instance_annotations(inst_mask, sem_pred)
            # Process ground truth annotations using the provided instance mask.
            gt_sem = gt_sem_masks[i].cpu().numpy()
            gt_inst = gt_inst_masks[i].cpu().numpy()
            gt_instances = get_instance_annotations(gt_inst, gt_sem)
            # Keep only the top max_queries predictions (by score).
            pred_instances = sorted(pred_instances, key=lambda x: x["score"], reverse=True)[:max_queries]
            for inst in pred_instances:
                inst["image_id"] = image_id
            for inst in gt_instances:
                inst["image_id"] = image_id
            all_predictions.extend(pred_instances)
            all_gt.extend(gt_instances)
            image_ids.append(image_id)
    gt_coco = {
        "images": [{"id": img_id} for img_id in image_ids],
        "annotations": [],
        "categories": [{"id": i, "name": f"cat_{i}"} for i in range(1, len(train_dataset.cat2label)+1)]
    }
    ann_id = 1
    for inst in all_gt:
        inst["id"] = ann_id
        ann_id += 1
        gt_coco["annotations"].append(inst)
    with open("temp_gt.json", "w") as f:
        json.dump(gt_coco, f)
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    coco_gt = COCO("temp_gt.json")
    coco_dt = coco_gt.loadRes(all_predictions)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='segm')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats

# Evaluate for AP@50, AP@100, AP@150, and AP@200
eval_log = open("evaluation_log.txt", "w")
for queries in [50, 100, 150, 200]:
    print(f"\nEvaluation with {queries} queries:")
    eval_log.write(f"Evaluation with {queries} queries:\n")
    stats = evaluate_instances(model, train_loader, device1, max_queries=queries)
    print("AP stats:", stats)
    eval_log.write("AP stats: " + str(stats) + "\n\n")
eval_log.close()
