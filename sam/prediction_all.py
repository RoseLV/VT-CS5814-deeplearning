import argparse
import os
import numpy as np
import torch
import cv2
# import matplotlib.pyplot as plt
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

def read_image(image_path, mask_path):
    img = cv2.imread(image_path)[..., ::-1]
    mask = cv2.imread(mask_path, 0)
    mask = (mask > 0).astype(np.uint8)
    return img, mask

def get_points(mask, num_points):
    points = []
    coords = np.argwhere(mask > 0)
    for _ in range(num_points):
        yx = coords[np.random.randint(len(coords))]
        points.append([[yx[1], yx[0]]])
    return np.array(points)

def calculate_iou(pred_mask, true_mask):
    pred_mask = (pred_mask > 0).astype(np.uint8)
    true_mask = (true_mask > 0).astype(np.uint8)
    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()
    return 1.0 if union == 0 and intersection == 0 else intersection / union

def process_image(image_path, mask_path, predictor, num_samples, output_dir):
    image, mask = read_image(image_path, mask_path)

    # 检查 mask 是否为空
    if mask.sum() == 0:
        print(f"跳过 {os.path.basename(image_path)}：mask 为空。")
        return

    input_points = get_points(mask, num_samples)

    # 如果没有采样到点，则跳过该图像
    if input_points.size == 0:
        print(f"跳过 {os.path.basename(image_path)}：未从 mask 中采样到点。")
        return

    image = np.ascontiguousarray(image)

    with torch.no_grad():
        predictor.set_image(image)
        masks, scores, logits = predictor.predict(
            point_coords=input_points,
            point_labels=np.ones([input_points.shape[0], 1])
        )

    masks = masks[:, 0].astype(bool)
    sorted_masks = masks[np.argsort(scores[:, 0])][::-1]
    seg_map = np.zeros_like(sorted_masks[0], dtype=np.uint8)
    occupancy_mask = np.zeros_like(sorted_masks[0], dtype=bool)

    for i in range(sorted_masks.shape[0]):
        mask = sorted_masks[i]
        if (mask * occupancy_mask).sum() / mask.sum() > 0.15:
            continue
        mask[occupancy_mask] = 0
        seg_map[mask] = i + 1
        occupancy_mask[mask] = 1

    rgb_image = np.zeros((seg_map.shape[0], seg_map.shape[1], 3), dtype=np.uint8)
    for id_class in range(1, seg_map.max() + 1):
        rgb_image[seg_map == id_class] = [np.random.randint(255) for _ in range(3)]

    annotation_path = os.path.join(output_dir, os.path.basename(mask_path).replace("_mask", "_annotation"))
    rgb_image_rgb = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(annotation_path, rgb_image_rgb)

    pred_mask = (cv2.imread(annotation_path, 0) > 0).astype(np.uint8)
    true_mask = (cv2.imread(mask_path, 0) > 0).astype(np.uint8)
    iou_score = calculate_iou(pred_mask, true_mask)
    print(f"处理完成 {os.path.basename(image_path)}：IOU = {iou_score:.4f}")


def main(image_dir, mask_dir, output_dir, sam2_checkpoint, model_cfg, num_samples):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有图片和 mask 文件的路径
    image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.tif')])
    mask_paths = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('.tif')])

    # 确保图片和 mask 数量一致
    if len(image_paths) != len(mask_paths):
        raise ValueError("Image and mask directories must contain the same number of files.")

    # 初始化模型
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")
    predictor = SAM2ImagePredictor(sam2_model)
    predictor.model.load_state_dict(torch.load("model.torch"))

    # 日志文件路径
    log_path = os.path.join(output_dir, "process_log.txt")
    iou_values = []  # 保存所有图像的 IOU 值

    with open(log_path, "w") as log_file:
        for image_path, mask_path in zip(image_paths, mask_paths):
            try:
                # 处理单张图片
                image_name = os.path.basename(image_path)
                process_image(image_path, mask_path, predictor, num_samples, output_dir)

                # 加载保存的预测 mask 和真实 mask
                annotation_path = os.path.join(output_dir, image_name.replace(".tif", "_annotation.tif"))
                pred_mask = (cv2.imread(annotation_path, 0) > 0).astype(np.uint8)
                true_mask = (cv2.imread(mask_path, 0) > 0).astype(np.uint8)

                # 计算 IOU
                iou_score = calculate_iou(pred_mask, true_mask)
                iou_values.append(iou_score)

                # 记录日志
                log_message = f"{image_name}: IOU = {iou_score:.4f}\n"
                print(log_message.strip())  # 打印到控制台
                log_file.write(log_message)  # 写入日志文件

            except Exception as e:
                error_message = f"Error processing {image_path}: {str(e)}\n"
                print(error_message.strip())
                log_file.write(error_message)

    # 计算平均 IOU
    if iou_values:
        average_iou = np.mean(iou_values)
        print(f"Average IOU: {average_iou:.4f}")
        with open(log_path, "a") as log_file:
            log_file.write(f"\nAverage IOU: {average_iou:.4f}\n")
    else:
        print("No IOU values calculated.")
        with open(log_path, "a") as log_file:
            log_file.write("\nNo IOU values calculated.\n")

# python prediction_all.py     --image_dir "/home/ran/data/deeplearning/Datasets/imagesTs"     --mask_dir "/home/ran/data/deeplearning/Datasets/labelsTs"     --output_dir "/home/ran/data/deeplearning/Datasets/sam_finetune_output"     --sam2_checkpoint "sam2_hiera_large.pt"     --model_cfg "sam2_hiera_l.yaml"     --num_samples 30

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch process images and masks using SAM.")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to the directory containing images.")
    parser.add_argument("--mask_dir", type=str, required=True, help="Path to the directory containing masks.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save annotations.")
    parser.add_argument("--sam2_checkpoint", type=str, required=True, help="Path to the SAM2 checkpoint file.")
    parser.add_argument("--model_cfg", type=str, required=True, help="Path to the SAM2 model config file.")
    parser.add_argument("--num_samples", type=int, default=30, help="Number of points to sample per mask.")
    args = parser.parse_args()

    main(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        output_dir=args.output_dir,
        sam2_checkpoint=args.sam2_checkpoint,
        model_cfg=args.model_cfg,
        num_samples=args.num_samples
    )
