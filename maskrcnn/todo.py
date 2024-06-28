# import os
# import json
# import torch
# import numpy as np
# from PIL import Image
# import torchvision.transforms as T
# from torchvision.transforms import functional as F
# from torch.utils.data import Dataset, DataLoader, Subset
# import torch.optim as optim
# from torchvision.models.detection import maskrcnn_resnet50_fpn
# from torchvision import transforms
# device = "cuda" if torch.cuda.is_available() else 'cpu'

# class CustomDataset(Dataset):
#     def __init__(self, root, annotation_file, transforms=None):
#         self.root = root
#         self.transforms = transforms
#         self.annotation_file = annotation_file

#         with open(annotation_file) as f:
#             self.coco = json.load(f)

#         self.imgs = {img['id']: img for img in self.coco['images']}
#         self.annotations = {img_id: [] for img_id in self.imgs.keys()}

#         for ann in self.coco['annotations']:
#             self.annotations[ann['image_id']].append(ann)

#     def __getitem__(self, idx):
#         img_id = list(self.imgs.keys())[idx]
#         img_info = self.imgs[img_id]
#         img_path = os.path.join(self.root, img_info['file_name'])

#         img = Image.open(img_path).convert("RGB")

#         annotations = self.annotations[img_id]
#         boxes = []
#         labels = []
#         masks = []

#         for ann in annotations:
#             xmin, ymin, width, height = ann['bbox']
#             boxes.append([xmin, ymin, xmin + width, ymin + height])
#             labels.append(ann['category_id'])

#             if 'segmentation' in ann and ann['segmentation']:
#                 mask = self.process_segmentation(ann['segmentation'], img_info)
#                 if mask is not None:
#                     masks.append(mask)

#         boxes = torch.as_tensor(boxes, dtype=torch.float32)
#         labels = torch.as_tensor(labels, dtype=torch.int64)

#         if masks:
#             masks = torch.as_tensor(np.stack(masks, axis=0), dtype=torch.uint8)
#         else:
#             masks = torch.empty((0, 0, 0), dtype=torch.uint8)

#         target = {}
#         target["boxes"] = boxes
#         target["labels"] = labels
#         target["masks"] = masks

#         if self.transforms:
#             img = self.transforms(img)

#         return img, target

#     def process_segmentation(self, segmentation, img_info):
#         if isinstance(segmentation, list) and isinstance(segmentation[0], list):
#             mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
#             for polygon in segmentation:
#                 coords = np.array(polygon).reshape((-1, 2))
#                 mask = self.draw_polygon(mask, coords)
#             return mask
#         elif isinstance(segmentation, str):
#             mask_path = os.path.join(self.root, segmentation)
#             if os.path.exists(mask_path):
#                 mask = Image.open(mask_path).convert("L")
#                 mask = np.array(mask)
#                 return mask
#         return None

#     def draw_polygon(self, mask, coords):
#         from skimage.draw import polygon
#         rr, cc = polygon(coords[:, 1], coords[:, 0], mask.shape)
#         mask[rr, cc] = 1
#         return mask

#     def __len__(self):
#         return len(self.imgs)

# train_root = r"C:\Users\pc\Downloads\Shampoo_5class.v2-only_non_defective-21-06-2024.coco\train"
# val_root = r"C:\Users\pc\Downloads\Shampoo_5class.v2-only_non_defective-21-06-2024.coco\valid"
# test_root = r"C:\Users\pc\Downloads\Shampoo_5class.v2-only_non_defective-21-06-2024.coco\test"

# train_annotation_file = r"C:\Users\pc\Downloads\Shampoo_5class.v2-only_non_defective-21-06-2024.coco\train\_annotations.coco.json"
# val_annotation_file = r"C:\Users\pc\Downloads\Shampoo_5class.v2-only_non_defective-21-06-2024.coco\valid\_annotations.coco.json"
# test_annotation_file = r"C:\Users\pc\Downloads\Shampoo_5class.v2-only_non_defective-21-06-2024.coco\test\_annotations.coco.json"

# # Define the transformation
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

# # Create datasets
# train_dataset = CustomDataset(root=train_root, annotation_file=train_annotation_file, transforms=transform)
# val_dataset = CustomDataset(root=val_root, annotation_file=val_annotation_file, transforms=transform)
# test_dataset = CustomDataset(root=test_root, annotation_file=test_annotation_file, transforms=transform)

# # Define collate function
# def collate_fn(batch):
#     return tuple(zip(*batch))

# # Create DataLoaders
# train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
# val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)
# test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

# # Define DataLoader
# def collate_fn(batch):
#     return tuple(zip(*batch))

# train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
# val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)
# test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)
# model = maskrcnn_resnet50_fpn(pretrained=True,num_classes = 6)
# model.to(device)

# def train_model(model, train_loader, val_loader, optimizer, num_epochs=5):
#     model.train()
#     for epoch in range(num_epochs):
#         train_loss = 0.0
#         for images, targets in train_loader:
#             # Images are already tensors, just move to device
#             images = [image.to(device) for image in images]
#             targets = [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]

#             loss_dict = model(images, targets)
#             losses = sum(loss for loss in loss_dict.values())
#             print(losses)

#             optimizer.zero_grad()
#             losses.backward()
#             optimizer.step()

#             train_loss += losses.item()

#         train_loss /= len(train_loader)

#         # Validation (similar change here)
#         model.eval()
#         val_loss = 0.0
#         with torch.no_grad():
#             for images, targets in val_loader:
#                 # Images are already tensors, just move to device
#                 images = [image.to(device) for image in images]
#                 targets = [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]

#                 loss_dict = model(images, targets)
#                 losses = sum(loss for loss in loss_dict.values())

#                 val_loss += losses.item()

#         val_loss /= len(val_loader)

#         print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}")
#         model.train()

# # Define optimizer
# params = [p for p in model.parameters() if p.requires_grad]
# optimizer = optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# # Train the model
# train_model(model, train_loader,val_loader, optimizer)

# from torchvision.transforms import functional as F

# def get_prediction(model, image_path, threshold):
#     model.eval()
#     img = Image.open(image_path).convert("RGB")
#     img = F.to_tensor(img).unsqueeze(0).to(device)

#     with torch.no_grad():
#         prediction = model(img)

#     pred_score = list(prediction[0]['scores'].detach().cpu().numpy())
#     pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
#     masks = (prediction[0]['masks']>0.5).squeeze().detach().cpu().numpy()
#     pred_class = [i for i in prediction[0]['labels'].detach().cpu().numpy()]
#     pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in prediction[0]['boxes'].detach().cpu().numpy()]
#     masks = masks[:pred_t+1]
#     pred_class = pred_class[:pred_t+1]
#     pred_boxes = pred_boxes[:pred_t+1]

#     return masks, pred_boxes, pred_class
