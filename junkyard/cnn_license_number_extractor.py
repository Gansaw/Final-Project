import torch
import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, MaskRCNN_ResNet50_FPN_Weights
from torchvision.transforms import functional as F
from PIL import Image
import os

from torchvision.datasets import CocoDetection
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

# 학습에 사용할 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 데이터 전처리 및 변환 설정
transform = transforms.Compose([
    transforms.ToTensor(),
])

# 데이터셋 경로
data_dir = os.path.join("coco_json_folder/train/")

# COCO 데이터셋 로드
dataset = CocoDetection(root=data_dir, annFile=os.path.join(data_dir, "_annotations.coco.json"), transform=transform)

# 데이터 로더 설정
data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

# Fast R-CNN 모델 초기화
fast_rcnn_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
fast_rcnn_model.to(device)

# Mask R-CNN 모델 초기화
mask_rcnn_model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
mask_rcnn_model.to(device)

# 옵티마이저 설정
optimizer = torch.optim.SGD([
    {'params': fast_rcnn_model.parameters()},
    {'params': mask_rcnn_model.parameters()}
], lr=0.005, momentum=0.9)

num_epochs = 10

# 손실 기록용 리스트
fast_rcnn_losses = []  # Fast R-CNN 손실 값 기록
mask_rcnn_losses = []  # Mask R-CNN 손실 값 기록

# 학습 루프
for epoch in range(num_epochs):
    epoch_fast_loss = 0.0
    epoch_mask_loss = 0.0
    for images, targets in data_loader:
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets if "boxes" in t]

        if not targets:
            continue
        
        optimizer.zero_grad()

        fast_loss_dict = fast_rcnn_model(images, targets)
        fast_losses = sum(loss for loss in fast_loss_dict.values())

        mask_loss_dict = mask_rcnn_model(images, targets)
        mask_losses = sum(loss for loss in mask_loss_dict.values())

        total_loss = fast_losses + mask_losses
        total_loss.backward()
        
        optimizer.step()

        epoch_fast_loss += fast_losses.item()
        epoch_mask_loss += mask_losses.item()

    avg_fast_loss = epoch_fast_loss / len(data_loader)
    avg_mask_loss = epoch_mask_loss / len(data_loader)

    fast_rcnn_losses.append(avg_fast_loss)
    mask_rcnn_losses.append(avg_mask_loss)

    print(f"Epoch [{epoch+1}/{num_epochs}], "
          f"Fast RCNN Loss: {avg_fast_loss:.4f}, "
          f"Mask RCNN Loss: {avg_mask_loss:.4f}")

# 학습이 완료된 후 손실 값의 변화를 그래프로 시각화
import matplotlib.pyplot as plt

plt.plot(range(1, num_epochs+1), fast_rcnn_losses, label='Fast RCNN Loss')
plt.plot(range(1, num_epochs+1), mask_rcnn_losses, label='Mask RCNN Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Losses')
plt.show()
