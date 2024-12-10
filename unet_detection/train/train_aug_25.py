# 80% von 50% (Gesamt) = 40%

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import random
import copy

# Check if CUDA is available and set the device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define U-Net Model
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()

        def conv_block(in_channels, out_channels):
            block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
            return block

        def up_conv(in_channels, out_channels):
            return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

        # Down-sampling layers
        self.enc1 = conv_block(in_channels, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.enc4 = conv_block(256, 512)

        # Max Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck layer
        self.bottleneck = conv_block(512, 1024)

        # Up-sampling layers
        self.upconv4 = up_conv(1024, 512)
        self.dec4 = conv_block(1024, 512)

        self.upconv3 = up_conv(512, 256)
        self.dec3 = conv_block(512, 256)

        self.upconv2 = up_conv(256, 128)
        self.dec2 = conv_block(256, 128)

        self.upconv1 = up_conv(128, 64)
        self.dec1 = conv_block(128, 64)

        # Output layer
        self.conv_last = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))

        # Decoder
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dec4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)

        return torch.sigmoid(self.conv_last(dec1))

# Define the custom Dataset class with logic to use only 50% of aug2 data
class CombinedSegmentationDataset(Dataset):
    def __init__(self, image_dirs, mask_dirs, transform=None, use_aug=False, aug_ratio=1.0):
        """
        Args:
            image_dirs (list): List of directories for images.
            mask_dirs (list): List of directories for masks.
            transform (callable, optional): Optional transform to be applied on a sample.
            use_aug (bool): Flag to indicate whether to use augmented data.
            aug_ratio (float): Ratio of augmented data to use (1.0 for all, 0.5 for half, etc.)
        """
        print("Aug Ratio=", aug_ratio)
        self.image_paths = []
        self.mask_paths = []
        self.transform = transform

        # Define image and mask directories based on augmentation flag
        if use_aug:
            self.image_dirs = image_dirs
            self.mask_dirs = mask_dirs
        else:
            # Use only non-augmented directories for validation
            self.image_dirs = [image_dirs[0], image_dirs[2]]  # Non-augmented directories for images
            self.mask_dirs = [mask_dirs[0], mask_dirs[2]]  # Non-augmented directories for masks

        # Collect image and mask paths
        for image_dir, mask_dir in zip(self.image_dirs, self.mask_dirs):
            image_files = sorted(os.listdir(image_dir))  # Ensure files are sorted
            mask_files = sorted(os.listdir(mask_dir))  # Corresponding mask files
            # If using aug2 data, only use a subset based on aug_ratio
            if "aug2" in image_dir or "aug2" in mask_dir:
                num_files = int(len(image_files) * aug_ratio)  # Use a subset of files
                selected_indices = random.sample(range(len(image_files)), num_files)  # Randomly select a subset
                image_files = [image_files[i] for i in selected_indices]
                mask_files = [mask_files[i] for i in selected_indices]
            
            for img_file, mask_file in zip(image_files, mask_files):
                self.image_paths.append(os.path.join(image_dir, img_file))
                self.mask_paths.append(os.path.join(mask_dir, mask_file))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image and mask
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = Image.open(image_path).convert('L')  # Convert to grayscale
        mask = Image.open(mask_path).convert('L')  # Binary mask in grayscale

        if self.transform is not None:
            image = self.transform(image)
            mask = self.transform(mask)

        mask = torch.where(mask > 0, torch.tensor(1.0), torch.tensor(0.0))  # Normalize mask to 0 and 1

        return image, mask


# IoU and Dice coefficient functions
def calculate_iou(pred, target):
    intersection = torch.sum(pred * target)
    union = torch.sum(pred) + torch.sum(target) - intersection
    return (intersection + 1e-6) / (union + 1e-6)

def calculate_dice(pred, target):
    intersection = torch.sum(pred * target)
    return (2 * intersection + 1e-6) / (torch.sum(pred) + torch.sum(target) + 1e-6)

# Validation loop
def validate_unet(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    iou_total = 0.0
    dice_total = 0.0

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            running_loss += loss.item() * images.size(0)

            # Convert outputs and masks to binary
            outputs = torch.where(outputs > 0.5, torch.tensor(1.0).to(device), torch.tensor(0.0).to(device))

            # Calculate IoU and Dice
            iou_total += calculate_iou(outputs, masks).item()
            dice_total += calculate_dice(outputs, masks).item()

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_iou = iou_total / len(dataloader)
    epoch_dice = dice_total / len(dataloader)

    return epoch_loss, epoch_iou, epoch_dice

# Training loop with early stopping and model saving
def train_unet(model, train_loader, val_loader, criterion, optimizer, num_epochs, patience, save_path):
    best_loss = float('inf')
    epochs_no_improve = 0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        running_loss = 0.0
        iou_total = 0.0
        dice_total = 0.0
        model.train()

        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            # Convert outputs and masks to binary
            outputs = torch.where(outputs > 0.5, torch.tensor(1.0).to(device), torch.tensor(0.0).to(device))

            # Calculate IoU and Dice
            iou_total += calculate_iou(outputs, masks).item()
            dice_total += calculate_dice(outputs, masks).item()

        train_loss = running_loss / len(train_loader.dataset)
        train_iou = iou_total / len(train_loader)
        train_dice = dice_total / len(train_loader)

        # Validation
        val_loss, val_iou, val_dice = validate_unet(model, val_loader, criterion)

        print(f'Train Loss: {train_loss:.4f}, IoU: {train_iou:.4f}, Dice: {train_dice:.4f}')
        print(f'Val Loss: {val_loss:.4f}, IoU: {val_iou:.4f}, Dice: {val_dice:.4f}')

        # Check if we need to stop training early
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), save_path)
            print(f"Model improved and saved to {save_path}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"No improvement in loss for {epochs_no_improve} epoch(s).")

        # Early stopping
        if epochs_no_improve == patience:
            print("Early stopping!")
            break

    # Load best model weights
    model.load_state_dict(best_model_wts)

# Directory paths
image_dirs = [
    '/data/train/rgb_total',
    '/data/aug2/rgb_total',
    '/data/train/rgb_semi',
    '/data/aug2/rgb_semi'
]

mask_dirs = [
    '/data/train/gt_total',
    '/data/aug2/gt_total',
    '/data/train/gt_semi',
    '/data/aug2/gt_semi'
]

save_path = '/unet_detection/models/bestmodel_25.pth'

# Hyperparameters
batch_size = 4
num_epochs = 50
learning_rate = 1e-4
patience = 3  # Early stopping patience

# Data transformations
transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor()
])

# Create datasets and dataloaders
train_dataset = CombinedSegmentationDataset(
    image_dirs=image_dirs, 
    mask_dirs=mask_dirs, 
    transform=transform, 
    use_aug=True, 
    aug_ratio=0.5  # Use 50% of the augmented data to 25%
)
val_dataset = CombinedSegmentationDataset(
    image_dirs=image_dirs, 
    mask_dirs=mask_dirs, 
    transform=transform, 
    use_aug=False
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Initialize model, loss function, and optimizer
model = UNet(in_channels=1, out_channels=1).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
train_unet(model, train_loader, val_loader, criterion, optimizer, num_epochs, patience, save_path)
