import numpy as np
import torch
import cv2
import os
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import matplotlib.pyplot as plt

# 初始化损失历史
loss_history = []
val_loss_history = []
val_iou_history = []

# Paths
data_dir_tr = "/home/ran/data/deeplearning/Datasets/"
image_dir = os.path.join(data_dir_tr, "imagesTrRgb")
label_dir = os.path.join(data_dir_tr, "labelsTr")

# Ensure both subdirectories exist
assert os.path.exists(image_dir), f"Image directory does not exist: {image_dir}"
assert os.path.exists(label_dir), f"Label directory does not exist: {label_dir}"

# List to store dataset entries
data = []

# Load data
for name in os.listdir(image_dir):
    if name.endswith(".tif"):
        image_path = os.path.join(image_dir, name)
        label_path = os.path.join(label_dir, name[:-4] + ".tif")
        if os.path.exists(label_path):
            data.append({"image": image_path, "annotation": label_path})
        else:
            print(f"Warning: Label not found for image: {name}")

print(f"\nTotal dataset size: {len(data)}")

# Split data into training and validation sets
np.random.seed(42)
indices = np.arange(len(data))
# np.random.shuffle(indices)

split_point = int(len(data) * 0.9)
train_data = [data[i] for i in indices[:split_point]]
val_data = [data[i] for i in indices[split_point:]]

print(f"Training dataset size: {len(train_data)}")
print(f"Validation dataset size: {len(val_data)}")

# Read batch function
def read_batch(data):
    ent = data[np.random.randint(len(data))]
    Img = cv2.imread(ent["image"])[..., ::-1]
    ann_map = cv2.imread(ent["annotation"], cv2.IMREAD_GRAYSCALE)
    ann_map = (ann_map > 0).astype(np.uint8)

    inds = np.unique(ann_map)[1:]
    points = []
    masks = []
    print(inds)
    for ind in inds:
        mask = (ann_map == ind).astype(np.uint8)
        masks.append(mask)
        coords = np.argwhere(mask > 0)
        yx = np.array(coords[np.random.randint(len(coords))])
        points.append([[yx[1], yx[0]]])

    return Img, np.array(masks), np.array(points), np.ones([len(masks), 1])


def read_batch(data, num_points=10):
    ent = data[np.random.randint(len(data))]
    Img = cv2.imread(ent["image"])[..., ::-1]
    ann_map = cv2.imread(ent["annotation"], cv2.IMREAD_GRAYSCALE)
    ann_map = (ann_map > 0).astype(np.uint8)

    inds = np.unique(ann_map)[1:]  # 获取所有有效的掩膜区域索引（排除背景）
    points = []
    masks = []

    for ind in inds:
        mask = (ann_map == ind).astype(np.uint8)  # 当前掩膜区域
        masks.append(mask)

        coords = np.argwhere(mask > 0)  # 获取该掩膜区域的所有像素坐标
        sampled_indices = np.random.choice(len(coords), min(num_points, len(coords)), replace=False)  # 随机采样点
        sampled_coords = coords[sampled_indices]  # 获取采样坐标

        for yx in sampled_coords:
            points.append([[yx[1], yx[0]]])  # 转换为 (x, y) 格式

    return Img, np.array(masks), np.array(points), np.ones([len(points), 1])

# Load model
sam2_checkpoint = "sam2_hiera_small.pt"
model_cfg = "sam2_hiera_s.yaml"
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")
predictor = SAM2ImagePredictor(sam2_model)

predictor.model.sam_mask_decoder.train(True)
predictor.model.sam_prompt_encoder.train(True)

optimizer = torch.optim.AdamW(params=predictor.model.parameters(), lr=1e-5, weight_decay=4e-5)
scaler = torch.cuda.amp.GradScaler()
mean_iou = 0

plt.figure(figsize=(10, 6))

# Early Stopping 参数
early_stop_patience = 20  # 连续多少次验证损失上涨后停止训练
early_stop_counter = 0    # 连续验证损失上涨的次数
best_val_loss = float('inf')  # 初始最优验证损失

# Training loop
for itr in range(100000):
    with torch.cuda.amp.autocast():
        image, mask, input_point, input_label = read_batch(train_data, num_points=10)
        print(f"Number of points sampled: {len(input_point)}")
        if mask.shape[0] == 0:
            continue
        image = image.copy()  # 确保数组是内存连续的
        predictor.set_image(image)

        mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(
            input_point, input_label, box=None, mask_logits=None, normalize_coords=True
        )
        sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
            points=(unnorm_coords, labels), boxes=None, masks=None
        )
        high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
        low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
            image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
            image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
            repeat_image=False,
            high_res_features=high_res_features,
        )
        prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])

        gt_mask = torch.tensor(mask.astype(np.float32)).cuda()
        prd_mask = torch.sigmoid(prd_masks[:, 0])
        seg_loss = (-gt_mask * torch.log(prd_mask + 1e-5) - (1 - gt_mask) * torch.log((1 - prd_mask) + 1e-5)).mean()

        inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
        iou = inter / (gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter)
        score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
        loss = seg_loss + score_loss * 0.05

        predictor.model.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_history.append(loss.item())
    # 每1000次保存模型（如果满足条件）
    if itr % 1000 == 0 and should_save_model:
        torch.save(predictor.model.state_dict(), "model.torch")
        print("Model saved.")
        plt.clf()
        plt.plot(loss_history, label="Training Loss")
        if len(val_loss_history) > 0:
            plt.plot(
                np.arange(500, 500 * (len(val_loss_history) + 1), 500),
                val_loss_history,
                label="Validation Loss",
                linestyle="--",
            )
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig("loss_plot.png")
