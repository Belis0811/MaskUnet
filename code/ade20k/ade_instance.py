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
from sklearn.metrics import precision_score, recall_score, f1_score
import cv2
import json
from panopticapi.utils import rgb2id

# -------------------------------
# Parameters and dataset paths
# -------------------------------
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

# -------------------------------
# Semantic segmentation dataset (ADE20K)
# -------------------------------
class ADE20KSegmentationDataset(Dataset):
    def __init__(self, objectinfo_txt, img_dir, ann_dir, transforms=None, img_size=(128, 128)):
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.transforms = transforms
        self.img_width, self.img_height = img_size

        # Load categories from objectInfo150.txt
        with open(objectinfo_txt, 'r') as f:
            # Each non-empty line corresponds to one category name.
            categories = [line.strip() for line in f if line.strip()]
        self.categories = categories
        # For ADE20K the segmentation mask pixel values represent the category id.
        # Build a mapping from category id (here, simply the index) to label id.
        self.cat2label = {i: i for i in range(len(categories))}

        # List all image files (adjust extensions as needed)
        self.img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg') or f.endswith('.png')])
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        img_filename = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_filename)
        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            raise FileNotFoundError(f"Could not find {img_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # Assume the annotation has the same base name with a .png extension
        ann_filename = os.path.splitext(img_filename)[0] + '.png'
        ann_path = os.path.join(self.ann_dir, ann_filename)
        mask = cv2.imread(ann_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Could not find {ann_path}")
        
        # Resize image and mask
        image_rgb = cv2.resize(image_rgb, (self.img_width, self.img_height), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.img_width, self.img_height), interpolation=cv2.INTER_NEAREST)
        
        if self.transforms:
            image_rgb = self.transforms(image_rgb)
        
        mask = torch.from_numpy(mask).long()
        return image_rgb, mask

objectinfo_path = os.path.join(ADEK_ROOT, 'objectInfo150.txt')

train_dataset = ADE20KSegmentationDataset(
    objectinfo_txt=objectinfo_path,
    img_dir=IMG_DIR_TRAIN,
    ann_dir=ANN_DIR_TRAIN,
    transforms=ToTensor(),
    img_size=(IMG_HEIGHT, IMG_WIDTH)
)

val_dataset = ADE20KSegmentationDataset(
    objectinfo_txt=objectinfo_path,
    img_dir=IMG_DIR_VAL,
    ann_dir=ANN_DIR_VAL,
    transforms=ToTensor(),
    img_size=(IMG_HEIGHT, IMG_WIDTH)
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

# -------------------------------
# maskUnet model (unchanged)
# -------------------------------
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

# -------------------------------
# Early stopping (unchanged)
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
            print(f'Saving model...')
        torch.save(model.state_dict(), 'checkpoint_ade_inst.pth')

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

# -------------------------------
# Set up model, loss, optimizer 
# -------------------------------
c_in = 3
c_out = len(train_dataset.cat2label)
print(f'total number of classes used: {c_out}')
if torch.cuda.device_count() >= 1:
    device1 = torch.device("cuda:0")
    device2 = torch.device("cuda:1")
else:
    raise RuntimeError("Multiple GPUs are required.")

model = UNet(c_in, c_out)
# model_dict = model.state_dict()

# checkpoint = torch.load('checkpoint_ade_inst.pth')
# modified_state_dict = {key.replace('module.', ''): value for key, value in checkpoint.items()}
# # pretrained_dict = {k: v for k, v in modified_state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
# model.load_state_dict(modified_state_dict)
# model_dict.update(pretrained_dict)
# model.load_state_dict(model_dict)

model = torch.nn.DataParallel(model)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-1)
early_stopping = EarlyStopping(patience=10, verbose=True)

# -------------------------------
# Training loop (unchanged)
# -------------------------------
num_epochs = 1000
model.to(device1)
best_loss = float("inf")
log_file = open("training_log_adek_inst.txt", "w")
for epoch in range(0, num_epochs):
    model.train()
    total_loss = 0.0
    for i, (inputs, labels) in enumerate(tqdm(train_loader)):
        inputs, labels = inputs.to(device1), labels.to(device1)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        if i % 500 == 0 and i != 0:
            print(f"Epoch {epoch+1}: Batch[{i}/{len(train_loader)}] Loss: {total_loss / i}")
            if i % 200 == 0:
                print(f'Saving model...')
                torch.save(model.state_dict(), 'checkpoint_ade_inst.pth')
                
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss}")
    best_loss = min(best_loss, avg_loss)
    torch.save(model.state_dict(), 'epoch_ade_inst.pth')
    log_file.write(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss}\n")
    log_file.write(f'Best loss: {best_loss}\n\n')
    log_file.flush()
    if early_stopping(avg_loss, model):
        print('Early stopping triggered')
        break
print(f'Best loss is {best_loss}')
log_file.close()

# -------------------------------
# Evaluation for instance segmentation
# -------------------------------
# Here we post-process the predicted semantic masks into instance masks using connected-component analysis.
def mask_to_rle(binary_mask):
    import pycocotools.mask as maskUtils
    rle = maskUtils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    if isinstance(rle["counts"], bytes):
        rle["counts"] = rle["counts"].decode("utf-8")
    return rle

def get_instances_from_mask(mask, prob_map=None):
    """
    Convert a semantic mask (H x W) into a list of instance dictionaries.
    If a probability map (shape: [c, H, W]) is given, the average probability
    for the predicted label over the instance region is used as its score.
    """
    instances = []
    unique_labels = np.unique(mask)
    for label in unique_labels:
        if label == 0:  # assume background is 0
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

def evaluate_instances(model, data_loader, device, max_queries):
    model.eval()
    all_predictions = []
    all_gt = []
    image_ids = []
    for images, gt_masks in tqdm(data_loader, desc="Evaluating"):
        images = images.to(device)
        with torch.no_grad():
            outputs = model(images)  # outputs: [B, c, H, W]
            probs = torch.softmax(outputs/0.5, dim=1).cpu().numpy()  # [B, c, H, W]
            preds = np.argmax(probs, axis=1)  # [B, H, W]
        for i in range(len(images)):
            image_id = i  # assign an id per image (adjust as needed)
            pred_mask = preds[i]
            prob_map = probs[i]
            gt_mask = gt_masks[i].cpu().numpy()
            pred_instances = get_instances_from_mask(pred_mask, prob_map=prob_map)
            gt_instances = get_instances_from_mask(gt_mask, prob_map=None)
            # Keep only the top max_queries predictions (by score)
            pred_instances = sorted(pred_instances, key=lambda x: x["score"], reverse=True)[:max_queries]
            for inst in pred_instances:
                inst["image_id"] = image_id
            for inst in gt_instances:
                inst["image_id"] = image_id
            all_predictions.extend(pred_instances)
            all_gt.extend(gt_instances)
            image_ids.append(image_id)
    # Build a dummy COCO ground truth structure
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

# Open a log file to record evaluation results
eval_log = open("evaluation_log.txt", "w")
for queries in [50, 100, 150, 200]:
    print(f"\nEvaluation with {queries} queries:")
    eval_log.write(f"Evaluation with {queries} queries:\n")
    stats = evaluate_instances(model, val_loader, device1, max_queries=queries)
    print("AP stats:", stats)
    eval_log.write("AP stats: " + str(stats) + "\n\n")
eval_log.close()
