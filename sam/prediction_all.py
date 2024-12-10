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

    # Check if mask is empty
    if mask.sum() == 0:
        print(f"Skipping {os.path.basename(image_path)}: mask is empty.")
        return

    input_points = get_points(mask, num_samples)

    # Skip the image if no points are sampled
    if input_points.size == 0:
        print(f"Skipping {os.path.basename(image_path)}: no points sampled from the mask.")
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
    print(f"Processing complete {os.path.basename(image_path)}: IOU = {iou_score:.4f}")


def main(image_dir, mask_dir, output_dir, sam2_checkpoint, model_cfg, num_samples):
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get paths of all image and mask files
    image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.tif')])
    mask_paths = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('.tif')])

    # Ensure the number of images and masks match
    if len(image_paths) != len(mask_paths):
        raise ValueError("Image and mask directories must contain the same number of files.")

    # Initialize the model
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")
    predictor = SAM2ImagePredictor(sam2_model)
    predictor.model.load_state_dict(torch.load("model.torch"))

    # Log file path
    log_path = os.path.join(output_dir, "process_log.txt")
    iou_values = []  # Save IOU values for all images

    with open(log_path, "w") as log_file:
        for image_path, mask_path in zip(image_paths, mask_paths):
            try:
                # Process a single image
                image_name = os.path.basename(image_path)
                process_image(image_path, mask_path, predictor, num_samples, output_dir)

                # Load saved prediction mask and ground truth mask
                annotation_path = os.path.join(output_dir, image_name.replace(".tif", "_annotation.tif"))
                pred_mask = (cv2.imread(annotation_path, 0) > 0).astype(np.uint8)
                true_mask = (cv2.imread(mask_path, 0) > 0).astype(np.uint8)

                # Calculate IOU
                iou_score = calculate_iou(pred_mask, true_mask)
                iou_values.append(iou_score)

                # Log results
                log_message = f"{image_name}: IOU = {iou_score:.4f}\n"
                print(log_message.strip())  # Print to console
                log_file.write(log_message)  # Write to log file

            except Exception as e:
                error_message = f"Error processing {image_path}: {str(e)}\n"
                print(error_message.strip())
                log_file.write(error_message)

    # Calculate average IOU
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
