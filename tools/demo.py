#!/usr/bin/env python
# ------------------------------------------------------------------------------
# FiDIP Demo Script
# Copyright (c) 2025 Augmented Cognition Lab, Northeastern University
# Licensed under The Apache-2.0 License
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import sys
import cv2
import numpy as np
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

# Add lib to PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import project modules
import _init_paths
from lib.config import cfg
from lib.config import update_config
from lib.core.inference import get_final_preds
from lib.utils.transforms import get_affine_transform
import models

# COCO keypoint names
KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

# COCO skeleton definition (pairs of keypoint indices that form a line)
SKELETON = [
    [0, 1], [0, 2], [1, 3], [2, 4],  # Face
    [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],  # Upper body
    [5, 11], [6, 12], [11, 12],  # Torso
    [11, 13], [13, 15], [12, 14], [14, 16]  # Lower body
]

# Colors
KEYPOINT_COLOR = (0, 0, 255)  # Red in BGR
SKELETON_COLOR = (0, 255, 0)  # Green in BGR
TEXT_COLOR = (0, 0, 255)  # Blue in BGR

# Confidence threshold for displaying keypoints
CONFIDENCE_THRESHOLD = 0.3


def parse_args():
    parser = argparse.ArgumentParser(description='FiDIP Demo')
    
    parser.add_argument('--cfg', 
                        help='experiment configuration file name',
                        required=True, 
                        type=str)
    
    parser.add_argument('--image', 
                        help='image file path',
                        required=True, 
                        type=str)
    
    parser.add_argument('--model', 
                        help='model file path',
                        required=True, 
                        type=str)
    
    parser.add_argument('--output', 
                        help='output image file path',
                        default='output_pose.jpg', 
                        type=str)
    
    parser.add_argument('--conf-threshold',
                        help='confidence threshold for displaying keypoints',
                        default=0.3,
                        type=float)
    
    parser.add_argument('--output-size',
                        help='size of the output image (width,height)',
                        default='512,512',
                        type=str)
    
    # Required arguments for config system
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')
    
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    
    # Set confidence threshold
    global CONFIDENCE_THRESHOLD
    CONFIDENCE_THRESHOLD = args.conf_threshold
    
    # Override TEST.MODEL_FILE with the model argument
    if args.model:
        if 'TEST.MODEL_FILE' not in args.opts:
            args.opts.extend(['TEST.MODEL_FILE', args.model])
    
    update_config(cfg, args)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED
    
    # Load model
    print('=> Loading model...')
    model_p, model_d = eval('models.'+cfg.MODEL.NAME+'.get_adaptive_pose_net')(
        cfg, is_train=False
    )
    
    if args.model:
        print('=> Loading model from {}'.format(args.model))
        model_p.load_state_dict(torch.load(args.model), strict=False)
    
    model_p = torch.nn.DataParallel(model_p, device_ids=cfg.GPUS).cuda()
    model_p.eval()
    
    # Load image
    print('=> Loading image from {}'.format(args.image))
    image = cv2.imread(args.image, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        print("Error: Could not read image {}".format(args.image))
        return
    
    # Get image dimensions
    height, width, _ = image.shape
    
    # Prepare input for the model
    input_size = cfg.MODEL.IMAGE_SIZE
    
    # Center and scale for the full image
    center = np.array([width / 2.0, height / 2.0])
    scale = np.array([width / 200.0, height / 200.0])
    
    # Create transformation matrix
    trans = get_affine_transform(center, scale, 0, input_size)
    
    # Apply transformation to get model input
    input_img = cv2.warpAffine(
        image, 
        trans, 
        (int(input_size[1]), int(input_size[0])),
        flags=cv2.INTER_LINEAR
    )
    
    # Save the transformed image for visualization
    vis_image = input_img.copy()
    
    # Normalize and convert to tensor
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    
    input_tensor = transform(input_img).unsqueeze(0)
    
    # Run inference
    print('=> Running model inference...')
    with torch.no_grad():
        input_tensor = input_tensor.cuda()
        _, output = model_p(input_tensor)
        
        # Get predictions in the model's input space
        # Use a dummy center and scale since we'll be drawing directly on the transformed image
        dummy_center = np.array([input_size[1] / 2.0, input_size[0] / 2.0])
        dummy_scale = np.array([input_size[1] / 200.0, input_size[0] / 200.0])
        preds, maxvals = get_final_preds(
            cfg, output.clone().cpu().numpy(), np.array([dummy_center]), np.array([dummy_scale])
        )
    
    # Visualize results
    print("=> Visualizing results...")
    # Create a copy for visualization
    vis_image = input_img.copy()
    
    # Draw keypoints and skeleton
    keypoints = []
    for i, (point, conf) in enumerate(zip(preds[0], maxvals[0])):
        x, y = int(point[0]), int(point[1])
        confidence = conf[0]
        
        # Store valid keypoints
        if confidence > args.conf_threshold:
            keypoints.append((i, x, y, confidence))
            
            # Draw keypoint with size based on confidence
            radius = int(5 * min(1.0, confidence + 0.3))
            cv2.circle(vis_image, (x, y), radius, KEYPOINT_COLOR, -1)
    
    # Draw skeleton with thickness based on average confidence of endpoints
    for connection in SKELETON:
        idx1, idx2 = connection
        
        # Find keypoints by index
        kp1 = next((kp for kp in keypoints if kp[0] == idx1), None)
        kp2 = next((kp for kp in keypoints if kp[0] == idx2), None)
        
        # Draw line if both keypoints are valid
        if kp1 is not None and kp2 is not None:
            # Calculate line thickness based on average confidence
            avg_conf = (kp1[3] + kp2[3]) / 2.0
            thickness = max(1, int(3 * avg_conf))
            
            cv2.line(vis_image, (kp1[1], kp1[2]), (kp2[1], kp2[2]), SKELETON_COLOR, thickness)
    
    # Add legend for colors
    legend_x = 10
    legend_y = 10
    
    # Draw keypoint color legend
    cv2.circle(vis_image, (legend_x + 5, legend_y + 5), 5, KEYPOINT_COLOR, -1)
    cv2.putText(vis_image, "Keypoint", (legend_x + 15, legend_y + 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)
    
    # Draw skeleton color legend
    cv2.line(vis_image, (legend_x, legend_y + 20), (legend_x + 10, legend_y + 20), 
             SKELETON_COLOR, 2)
    cv2.putText(vis_image, "Skeleton", (legend_x + 15, legend_y + 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)
    
    # Parse output size
    try:
        output_width, output_height = map(int, args.output_size.split(','))
    except:
        output_width, output_height = 512, 512
        print(f"Warning: Could not parse output size '{args.output_size}', using default 512x512")
    
    # Resize the result image to the specified output size
    vis_image = cv2.resize(vis_image, (output_width, output_height), interpolation=cv2.INTER_AREA)
    
    # Save result
    print('=> Saving result to {}'.format(args.output))
    cv2.imwrite(args.output, vis_image)
    
    # Create a separate image with keypoint confidence values
    conf_img = np.ones((500, 300, 3), dtype=np.uint8) * 255
    
    # Add title
    cv2.putText(conf_img, "Keypoint Confidence Values", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Add confidence values
    y_pos = 60
    for i, _, _, conf in sorted(keypoints, key=lambda x: x[0]):
        # Color text based on confidence (red for low, green for high)
        if conf < 0.5:
            color = (0, 0, 255)  # Red for low confidence
        elif conf < 0.7:
            color = (0, 165, 255)  # Orange for medium confidence
        else:
            color = (0, 255, 0)  # Green for high confidence
            
        text = f"{KEYPOINT_NAMES[i]}: {conf:.2f}"
        cv2.putText(conf_img, text, (10, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        y_pos += 20
    
    # Save confidence image
    conf_output = os.path.splitext(args.output)[0] + '_conf.jpg'
    cv2.imwrite(conf_output, conf_img)
    print('=> Saving confidence values to {}'.format(conf_output))
    
    print('Done!')


if __name__ == '__main__':
    main()
