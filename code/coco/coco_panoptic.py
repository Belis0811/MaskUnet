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

BATCH_SIZE = 14
IMG_WIDTH = 128
IMG_HEIGHT = 128

TRAIN_PATH = '../../data/COCO/'
TEST_PATH = '../../data/COCO/'
warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42
class COCOSegmentationDataset(Dataset):
    def __init__(self, panoptic_json, panoptic_root, img_dir, transforms=None, img_size=(128, 128)):
        self.img_dir = img_dir
        self.panoptic_root = panoptic_root
        self.transforms = transforms
        self.img_width, self.img_height = img_size
        self.panoptic_json = panoptic_json
        
        with open(panoptic_json, 'r') as f:
            self.panoptic_data = json.load(f)

        self.images = self.panoptic_data["images"]

        self.annotations = {}
        for ann in self.panoptic_data["annotations"]:
            self.annotations[ann["image_id"]] = ann

        self.categories = self.panoptic_data["categories"]

        all_cat_ids = sorted(cat["id"] for cat in self.categories)
        self.cat2label = {cat_id: idx for idx, cat_id in enumerate(all_cat_ids)}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        image_id = img_info["id"]
        w, h = img_info["width"], img_info["height"]
        img_filename = img_info["file_name"]

        img_path = os.path.join(self.img_dir, img_filename)
        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            raise FileNotFoundError(f"Could not find {img_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        ann_info = self.annotations[image_id]
        seg_filename = ann_info["file_name"]
        seg_path = os.path.join(self.panoptic_root, seg_filename)
        seg_bgr = cv2.imread(seg_path, cv2.IMREAD_COLOR)
        if seg_bgr is None:
            raise FileNotFoundError(f"Could not find {seg_path}")

        seg_rgb = cv2.cvtColor(seg_bgr, cv2.COLOR_BGR2RGB)
        seg_id_map = rgb2id(seg_rgb)

        # Semantic Mask: Stores category labels
        semantic_mask = np.zeros((h, w), dtype=np.int32)
        # Instance Mask: Stores unique object IDs
        instance_mask = np.zeros((h, w), dtype=np.int32)

        for seg in ann_info["segments_info"]:
            cat_id = seg["category_id"]
            seg_id = seg["id"]
            label_id = self.cat2label[cat_id]

            mask_pixels = (seg_id_map == seg_id)
            semantic_mask[mask_pixels] = label_id
            instance_mask[mask_pixels] = seg_id  # Unique instance ID

        image_rgb = cv2.resize(image_rgb, (self.img_width, self.img_height), interpolation=cv2.INTER_LINEAR)
        semantic_mask = cv2.resize(semantic_mask, (self.img_width, self.img_height), interpolation=cv2.INTER_NEAREST)
        instance_mask = cv2.resize(instance_mask, (self.img_width, self.img_height), interpolation=cv2.INTER_NEAREST)

        if self.transforms:
            image_rgb = self.transforms(image_rgb)

        semantic_mask = torch.from_numpy(semantic_mask).long()
        instance_mask = torch.from_numpy(instance_mask).long()

        return image_rgb, semantic_mask, instance_mask

train_dataset = COCOSegmentationDataset(
    panoptic_json=os.path.join(TRAIN_PATH, 'annotations', 'panoptic_train_subset.json'),
    panoptic_root=os.path.join(TRAIN_PATH, 'panoptic_train_subset'), 
    img_dir=os.path.join(TRAIN_PATH, 'train_subset'),
    transforms=ToTensor(),
    img_size=(IMG_HEIGHT, IMG_WIDTH)
)
val_dataset = COCOSegmentationDataset(
    panoptic_json=os.path.join(TRAIN_PATH,  'annotations', 'panoptic_val2017.json'),
    panoptic_root=os.path.join(TRAIN_PATH, 'panoptic_val2017'), 
    img_dir=os.path.join(TRAIN_PATH, 'val2017'),
    transforms=ToTensor(),
    img_size=(IMG_HEIGHT, IMG_WIDTH)
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

import random
import matplotlib.pyplot as plt
import numpy as np

def visualize_random_sample(dataset):
    idx = random.randint(0, len(dataset) - 1)
    image, semantic_mask, instance_mask = dataset[idx]  
    
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
    plt.imshow(semantic_mask.cpu().numpy() if isinstance(semantic_mask, torch.Tensor) else semantic_mask)
    plt.title("Semantic Mask")

    plt.subplot(1, 3, 3)
    plt.imshow(instance_mask.cpu().numpy() if isinstance(instance_mask, torch.Tensor) else instance_mask)
    plt.title("Instance Mask")

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
        #print(x.shape)
        #print(skip_x.shape)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        #emb = self.emb_layer(t)[:, :, None, None].expand(-1, -1, x.shape[-2], x.shape[-1])
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
        # self.output_conv = nn.Conv2d(64, c_out, kernel_size=1)
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
        # output = self.output_conv(x)
        output = self.final_layer(x)
        return output

#define early stopping
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
        torch.save(model.state_dict(), 'checkpoint_pan.pth')

import cv2

def generate_instance_mask(semantic_mask):
    instance_mask = np.zeros_like(semantic_mask, dtype=np.int32)
    unique_classes = np.unique(semantic_mask)

    for class_id in unique_classes:
        if class_id == 0: 
            continue
        binary_mask = (semantic_mask == class_id).astype(np.uint8)
        num_labels, labels = cv2.connectedComponents(binary_mask)
        for label in range(1, num_labels):
            instance_mask[labels == label] = label 

    return instance_mask

import os
import cv2
import numpy as np
from panopticapi.utils import id2rgb

def save_predictions(image_id, segmentation_map):
    """ Save the predicted segmentation mask in COCO format """
    os.makedirs("predictions", exist_ok=True)  #Ensure folder exists
    save_path = f"predictions/{image_id}.png"

    # Convert instance mask to RGB format for COCO evaluation
    segmentation_rgb = id2rgb(segmentation_map)

    # Save the predicted mask
    cv2.imwrite(save_path, segmentation_rgb)

    return save_path

from panopticapi.evaluation import pq_compute
import json

def evaluate_panoptic(model, val_loader, dataset, device):
    model.eval()
    predictions = []
    
    os.makedirs("predictions", exist_ok=True) #Ensure predictions folder exists

    with torch.no_grad():
        for images, sem_masks, inst_masks in tqdm(val_loader):
            images = images.to(device)
            sem_logits = model(images)
            sem_preds = torch.argmax(sem_logits, dim=1).cpu().numpy()

            for i in range(len(images)):
                image_id = dataset.images[i]["id"]

                # Convert prediction to proper format
                pred_segmentation = sem_preds[i]

                # Save predicted segmentation mask
                pred_segmentation_path = save_predictions(image_id, pred_segmentation)

                predictions.append({
                    "image_id": image_id,
                    "file_name": f"{image_id}.png",
                    "segments_info": []  # You need to fill instance details
                })

    # Save JSON file
    pred_file = "predictions.json"
    with open(pred_file, "w") as f:
        json.dump({"annotations": predictions}, f)

    # Run Panoptic Quality computation
    pq_results = pq_compute(gt_json_file=dataset.panoptic_json, pred_json_file=pred_file)
    
    print(f"PQ: {pq_results['All']} | Things: {pq_results['Things']} | Stuff: {pq_results['Stuff']}")

from pycocotools.cocoeval import COCOeval

def compute_ap(gt_json, pred_json):
    coco_gt = COCO(gt_json)
    coco_pred = coco_gt.loadRes(pred_json)

    coco_eval = COCOeval(coco_gt, coco_pred, "segm")  
    coco_eval.params.iouThrs = np.linspace(0.30, 0.95, 10)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval.stats[0]


# from segmentation_models_pytorch import Unet

# pretrained_model = Unet(encoder_name="mit_b5", encoder_weights="imagenet",in_channels=3,classes=len(train_dataset.cat2label))
# pretrained_weights = pretrained_model.state_dict()
import torch.backends.cudnn as cudnn
#from torch_lr_finder import LRFinder

c_in = 3  # input channel 3 for RGB
c_out = len(train_dataset.cat2label)
print(f'total number of classes used: {c_out}')
if torch.cuda.device_count() >= 1:
    device1 = torch.device("cuda:0")
    device2 = torch.device("cuda:1")
else:
    raise RuntimeError("Multiple GPUs are required.")

model = UNet(c_in, c_out)
# checkpoint = torch.load('checkpoint_pan.pth')
# modified_state_dict = {key.replace('module.', ''): value for key, value in checkpoint.items()}

# model.load_state_dict(modified_state_dict)

cudnn.benchmark = True
model = torch.nn.DataParallel(model)


class InstanceContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
        self.loss_fn = nn.TripletMarginLoss(margin=self.margin)

    def forward(self, sem_mask, instance_mask):
        unique_instances = torch.unique(instance_mask)
        loss = 0.0
        count = 0

        for inst in unique_instances:
            if inst == 0:  # Ignore background
                continue

            inst_pixels = (instance_mask == inst).nonzero(as_tuple=True)
            if len(inst_pixels[0]) < 2:
                continue  # Ignore small instances

            # Select anchor and positive samples
            anchor = sem_mask[:, :, inst_pixels[0][0], inst_pixels[1][0]]
            positive = sem_mask[:, :, inst_pixels[0][1], inst_pixels[1][1]]

            # Correcting Negative Sample Selection
            negative_pixels = (instance_mask != inst).nonzero(as_tuple=True)
            if len(negative_pixels[0]) == 0:  
                continue  # If no negatives exist, skip this instance

            negative_idx = torch.randint(0, len(negative_pixels[0]), (1,))
            negative = sem_mask[:, :, negative_pixels[0][negative_idx], negative_pixels[1][negative_idx]]

            # Ensure same shape for triplet margin loss
            anchor = anchor.view(1, -1)
            positive = positive.view(1, -1)
            negative = negative.view(1, -1)

            loss += self.loss_fn(anchor, positive, negative)
            count += 1

        return loss / count if count > 0 else torch.tensor(0.0, device=sem_mask.device)
    
semantic_loss_fn = nn.CrossEntropyLoss()
instance_loss_fn = InstanceContrastiveLoss()



optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)



early_stopping = EarlyStopping(patience=10, verbose=True)

# Training loop
num_epochs = 1000
model.to(device1)
best_loss = float("inf")
best_iou = 0.0
log_file = open("training_log_panoptic.txt", "w")
for epoch in range(0,num_epochs):
    model.train()
    total_loss = 0.0
    total_iou = 0.0
    for i,(inputs,  semantic_labels, instance_labels) in enumerate(tqdm(train_loader)):
        inputs, semantic_labels, instance_labels = inputs.to(device1), semantic_labels.to(device1), instance_labels.to(device1)
        optimizer.zero_grad()
        semantic_logits = model(inputs)

        semantic_loss = semantic_loss_fn(semantic_logits, semantic_labels)
        instance_loss = instance_loss_fn(semantic_logits, instance_labels)
        
        loss = 0.9 * semantic_loss + 0.1 * instance_loss
        loss.backward()
        optimizer.step()
        
        iou = mean_iou(semantic_logits, semantic_labels, c_out)
        
        total_loss += loss.item()
        total_iou += iou

        if i % 5000 == 0 and i != 0:
            print(f"Epoch {epoch+1}: Batch[{i}/{len(train_loader)}] Loss: {total_loss / i} IoU: {total_iou / i}")
            if i % 2000 == 0:
                print(f'Saving model...')
                torch.save(model.state_dict(), 'checkpoint_pan.pth')
    avg_loss = total_loss / len(train_loader)
    avg_iou = total_iou / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss} IoU: {avg_iou}")

    best_loss = min(best_loss,avg_loss)
    best_iou = max(best_iou,avg_iou)
    
    if best_iou-avg_iou > 0.1:
        print("Model Overfit")
        break
    
    # scheduler.step()

    torch.save(model.state_dict(), 'epoch.pth')
    log_file.write(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss} IoU: {avg_iou}\n")
    log_file.write(f'Best loss: {best_loss}, Best IoU: {best_iou}\n\n')
    log_file.flush()

    if early_stopping(avg_loss, model):
        print('Early stopping triggered')
        break
    
    #if epoch % 1 == 0 and epoch != 0:
     #   pq_results = evaluate_panoptic(model, train_loader, train_dataset, device1)
      #  ap = compute_ap(train_dataset.panoptic_json, "predictions.json")
       # print(f"Validation PQ: {pq_results['All']} | AP: {ap}")
        #log_file.write(f"Epoch [{epoch+1}/{num_epochs}] Validation PQ: {pq_results['All']} | AP: {ap}\n\n")

    
# torch.save(model.state_dict(), 'model.pth')
print(f'Best loss is {best_loss}, best iou is {best_iou}')
log_file.close()

model = model.to(device1)
best_val_loss = float("inf")
best_val_iou = 0.0
log_file = open("validating_log_full.txt", "w")
model.eval()
criterion = nn.CrossEntropyLoss()
for epoch in range(0,num_epochs):
    
    total_val_loss = 0.0
    total_val_iou = 0.0
    num_batches = len(val_loader)

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader):
            inputs, labels = inputs.to(device1), labels.to(device1)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_val_iou += mean_iou(outputs, labels,c_out)
            total_val_loss += loss.item()
            
    preds_val = outputs

    avg_val_loss = total_val_loss / num_batches
    avg_val_iou = total_val_iou / num_batches

    print(f"Validation Loss: {avg_val_loss}, Validation IoU: {avg_val_iou}")
    
    best_val_loss = min(best_val_loss,avg_val_loss)
    best_val_iou = max(best_val_iou,avg_val_iou)
    
    log_file.write(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_val_loss} IoU: {avg_val_iou}\n")
    log_file.write(f'Best loss: {best_val_loss}, Best IoU: {best_val_iou}\n\n')
    log_file.flush()

print(f'Best Validation loss is {best_val_loss}, best Validation iou is {best_val_iou}')
log_file.close()

def visualize_predictions(model, dataset, device, idx=0):
    model.eval()
    with torch.no_grad():
        image, mask = dataset[idx] 
        image = image.unsqueeze(0).to(device)  
        
        output = model(image)
        print(output.shape)
        pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

        
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 3, 1)
        plt.imshow(image.squeeze(0).permute(1, 2, 0).cpu().numpy())
        plt.title("Original Image")

        plt.subplot(1, 3, 2)
        plt.imshow(mask)
        plt.title("Ground Truth")

        plt.subplot(1, 3, 3)
        plt.imshow(pred_mask)
        plt.title("Predicted Mask")

        plt.show()

test_idx = random.randint(0, len(val_dataset) - 1)
visualize_predictions(model, val_dataset, device1, idx=test_idx)
