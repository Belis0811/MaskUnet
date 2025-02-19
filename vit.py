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

BATCH_SIZE = 16
IMG_WIDTH = 128
IMG_HEIGHT = 128

TRAIN_PATH = './COCO/'
TEST_PATH = './COCO/'

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42
class COCOSegmentationDataset(Dataset):
    def __init__(self, panoptic_json, panoptic_root, img_dir, transforms=None, img_size=(128, 128)):
        self.img_dir = img_dir
        self.panoptic_root = panoptic_root
        self.transforms = transforms
        self.img_width, self.img_height = img_size
        
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

        semantic_mask = np.zeros((h, w), dtype=np.int32)

        for seg in ann_info["segments_info"]:
            cat_id = seg["category_id"]  
            seg_id = seg["id"]          

            label_id = self.cat2label[cat_id]

            mask_pixels = (seg_id_map == seg_id)
            semantic_mask[mask_pixels] = label_id

        image_rgb = cv2.resize(image_rgb, (self.img_width, self.img_height), interpolation=cv2.INTER_LINEAR)
        semantic_mask = cv2.resize(semantic_mask, (self.img_width, self.img_height), interpolation=cv2.INTER_NEAREST)

        if self.transforms:
            image_rgb = self.transforms(image_rgb)  

        semantic_mask = torch.from_numpy(semantic_mask).long()

        return image_rgb, semantic_mask
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
    image, mask = dataset[idx]
    if isinstance(image, torch.Tensor):
        # Permute to [H, W, C]
        image_np = image.permute(1, 2, 0).cpu().numpy()
        # If it's in range [0,1], scale for display:
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
    else:
        # If 'image' is already a NumPy array with shape [H, W, C]
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
import torch.nn.functional as F
from timm.models.vision_transformer import VisionTransformer, PatchEmbed


class CustomPatchEmbed(PatchEmbed):
    def __init__(self, img_size=(128, 128), patch_size=16, in_chans=3, embed_dim=768):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim
        )
        # Remove the internal checks on input dimensions:
        self.img_size = (128, 128)

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        x = self.proj(x)  # [B, embed_dim, H/patch_size, W/patch_size]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        if self.norm is not None:
            x = self.norm(x)
        return x

class CustomViT(VisionTransformer):
    def __init__(self, img_size=128, patch_size=16, in_chans=3, num_classes=768, **kwargs):
        super().__init__(
            img_size=224,       
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=num_classes,
            **kwargs
        )

        self.patch_size_value = patch_size
        
        self.patch_embed = CustomPatchEmbed(
            img_size=(img_size, img_size),
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=self.embed_dim
        )

    def forward(self, x):
        # x: [B, C, 128, 128]
        B, C, H, W = x.shape
        x = self.patch_embed(x)  # [B, num_patches, embed_dim]

        if self.pos_embed.shape[1] != x.shape[1]:
            pos_embed_2d = self.pos_embed[0, 1:, :].reshape(14, 14, self.embed_dim)
            pos_embed_2d = F.interpolate(
                pos_embed_2d.permute(2,0,1).unsqueeze(0),
                size=(int(H / self.patch_size_value), int(W / self.patch_size_value)),
                mode='bilinear',
                align_corners=False
            )
            new_pos_embed = pos_embed_2d.squeeze(0).permute(1,2,0).view(-1, self.embed_dim)
            # keep class token
            cls_token = self.pos_embed[0, 0, :].unsqueeze(0)
            new_pos_embed = torch.cat((cls_token, new_pos_embed), dim=0).unsqueeze(0)
            self.pos_embed = new_pos_embed.to(x.device)

        x = x + self.pos_embed[:, : x.size(1), :]
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

class ViTSegmentation(nn.Module):
    def __init__(self, num_classes):
        super(ViTSegmentation, self).__init__()
        # Our custom 128×128 ViT
        self.vit = CustomViT(
            img_size=128,
            patch_size=16,
            in_chans=3,
            num_classes=768,   # We keep embed_dim=768
            embed_dim=768,
            depth=12,
            num_heads=12,
        )
        # Simple decoder for segmentation
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(768, 256, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, num_classes, kernel_size=1)
        )
    
    def forward(self, x):
        B, C, H, W = x.shape
        # vit forward
        x = self.vit(x)  # [B, num_patches, 768]
        # Reshape to [B, 768, H/16, W/16]
        x = x.permute(0, 2, 1).view(B, 768, H // 16, W // 16)
        # decode
        x = self.decoder(x)
        # upsample to original
        x = torch.nn.functional.interpolate(
            x, size=(H, W), mode="bilinear", align_corners=False
        )
        return x

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
        torch.save(model.state_dict(), 'checkpoint_vit.pth')
# from segmentation_models_pytorch import Unet

# pretrained_model = Unet(encoder_name="mit_b5", encoder_weights="imagenet",in_channels=3,classes=len(train_dataset.cat2label))
# pretrained_weights = pretrained_model.state_dict()
import torch.backends.cudnn as cudnn

#print(f'total number of classes used: {c_out}')
if torch.cuda.device_count() > 1:
    device1 = torch.device("cuda:0")
    device2 = torch.device("cuda:1")
else:
    raise RuntimeError("Multiple GPUs are required.")

num_classes = len(train_dataset.cat2label)
model = ViTSegmentation(num_classes)
#pretrained_weights = "hf-hub:timm/vit_base_patch16_224"  
#model.vit.load_state_dict(torch.hub.load_state_dict_from_url(pretrained_weights, map_location=device1), strict=False)
#print("Pretrained segmentation weights loaded from timm.")
# checkpoint = torch.load('checkpoint.pth')
# modified_state_dict = {key.replace('module.', ''): value for key, value in checkpoint.items()}
# model.load_state_dict(modified_state_dict)

# def map_pretrained_to_check(pretrained_dict, custom_dict):
#     mapped_weights = {}
#     for key in pretrained_dict:
#         if key.startswith("encoder.conv1"):
#             new_key = key.replace("encoder.conv1", "initial_conv.conv_block.0")
#             if new_key in custom_dict:
#                 mapped_weights[new_key] = pretrained_dict[key]
#     return mapped_weights

# mapped_weights = map_pretrained_to_check(pretrained_weights, modified_state_dict)
# modified_state_dict.update(mapped_weights)
    
# model.load_state_dict(modified_state_dict)

# for name, param in model.named_parameters():
#     if name not in mapped_weights:
#         # print(f"Initializing unmatched layer: {name}")
#         if param.dim() >= 2: 
#             torch.nn.init.xavier_uniform_(param)
#         elif param.dim() == 1:  
#             torch.nn.init.zeros_(param)  


cudnn.benchmark = True
model = torch.nn.DataParallel(model)

# class DiceLoss(nn.Module):
#     def __init__(self, smooth=1.0):
#         super(DiceLoss, self).__init__()
#         self.smooth = smooth

#     def forward(self, pred, target):
#         num_classes = pred.shape[1]

#         target = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()

#         pred = pred.sigmoid()
#         # pred = pred.flatten(1)
#         # target = target.flatten(1)

#         intersection = (pred * target).sum(-1)
#         union = pred.sum(-1) + target.sum(-1)

#         dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
#         return 1 - dice_score.mean()
    
criterion = nn.CrossEntropyLoss()
# criterion = DiceLoss()

# optimizer = optim.Adam(model.parameters(), lr=1e-03)
optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)

# lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
# lr_finder.range_test(train_loader, end_lr=1, num_iter=100)

# lr_finder.plot()

# lr_finder.reset()

early_stopping = EarlyStopping(patience=10, verbose=True)
# scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=5e-4, steps_per_epoch=len(train_loader), epochs=400)

# Training loop
num_epochs = 1000
model.to(device1)
best_loss = float("inf")
best_iou = 0.0
log_file = open("training_log_full_4.txt", "w")
for epoch in range(0,num_epochs):
    model.train()
    total_loss = 0.0
    total_iou = 0.0
    for i,(inputs, labels) in enumerate(tqdm(train_loader)):
        inputs, labels = inputs.to(device1), labels.to(device1)
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        iou = mean_iou(outputs, labels, c_out)
        
        total_loss += loss.item()
        total_iou += iou

        if i % 5000 == 0 and i != 0:
            print(f"Epoch {epoch+1}: Batch[{i}/{len(train_loader)}] Loss: {total_loss / i} IoU: {total_iou / i}")
            if i % 2000 == 0:
                print(f'Saving model...')
                torch.save(model.state_dict(), 'checkpoint_vit.pth')
    avg_loss = total_loss / len(train_loader)
    avg_iou = total_iou / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss} IoU: {avg_iou}")

    best_loss = min(best_loss,avg_loss)
    best_iou = max(best_iou,avg_iou)
    
    if best_iou-avg_iou > 0.1:
        print("Model Overfit")
        break
    
    # scheduler.step()

    log_file.write(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss} IoU: {avg_iou}\n")
    log_file.write(f'Best loss: {best_loss}, Best IoU: {best_iou}\n\n')
    log_file.flush()

    if early_stopping(avg_loss, model):
        print('Early stopping triggered')
        break
    
# torch.save(model.state_dict(), 'model.pth')
print(f'Best loss is {best_loss}, best iou is {best_iou}')
log_file.close()

model = model.to(device1)
best_val_loss = float("inf")
best_val_iou = 0.0
log_file = open("validating_log_full.txt", "w")
model.eval()

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
