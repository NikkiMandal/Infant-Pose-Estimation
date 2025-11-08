from __future__ import absolute_import, division, print_function

import argparse
import os
import sys
import time
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import _init_paths
from lib.config import cfg, update_config
from lib.core.inference import get_final_preds
from lib.utils.transforms import get_affine_transform
import models

KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

SKELETON = [
    [0, 1], [0, 2], [1, 3], [2, 4],              # face
    [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],     # upper body
    [5, 11], [6, 12], [11, 12],                  # torso
    [11, 13], [13, 15], [12, 14], [14, 16]       # lower body
]

KEYPOINT_COLOR = (0, 0, 255)   # BGR red
SKELETON_COLOR = (0, 255, 0)   # BGR green
TEXT_COLOR = (255, 0, 0)       # BGR blue
# ------------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='FiDIP Video Demo')

    parser.add_argument('--cfg', required=True, type=str,
                        help='experiment configuration file')

    parser.add_argument('--model', required=True, type=str,
                        help='path to model .pth (e.g., models/hrnet_fidip.pth)')

    parser.add_argument('--input', required=True, type=str,
                        help='input video path or webcam index (e.g., 0)')

    parser.add_argument('--output', default='output_pose_video.mp4', type=str,
                        help='output video path')

    parser.add_argument('--conf-threshold', default=0.3, type=float,
                        help='confidence threshold for keypoint rendering')

    parser.add_argument('--no-flip', action='store_true',
                        help='disable horizontal flip test (faster)')

    parser.add_argument('--modelDir', type=str, default='')
    parser.add_argument('--logDir', type=str, default='')
    parser.add_argument('--dataDir', type=str, default='')
    parser.add_argument('--prevModelDir', type=str, default='')

    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)

    return parser.parse_args()


def load_model(cfg, model_path):
    model_p, model_d = eval('models.' + cfg.MODEL.NAME + '.get_adaptive_pose_net')(cfg, is_train=False)

    state_dict = torch.load(model_path, map_location='cpu')
    model_p.load_state_dict(state_dict, strict=False)
    
    model_p = torch.nn.DataParallel(model_p, device_ids=cfg.GPUS).cuda()
    model_p.eval()
    return model_p


def draw_pose(vis_img, preds, maxvals, conf_thresh):
    """
    Draw keypoints & skeleton onto vis_img (already in the network input space).
    """
    h, w = vis_img.shape[:2]
  
    keypoints = []
    for i, (pt, conf) in enumerate(zip(preds[0], maxvals[0])):
        x, y = int(pt[0]), int(pt[1])
        c = float(conf[0])
        if c >= conf_thresh and 0 <= x < w and 0 <= y < h:
            keypoints.append((i, x, y, c))
            radius = int(5 * min(1.0, c + 0.3))
            cv2.circle(vis_img, (x, y), max(2, radius), KEYPOINT_COLOR, -1)

    for a, b in SKELETON:
        kp1 = next((kp for kp in keypoints if kp[0] == a), None)
        kp2 = next((kp for kp in keypoints if kp[0] == b), None)
        if kp1 is not None and kp2 is not None:
            avg_c = 0.5 * (kp1[3] + kp2[3])
            thickness = max(1, int(3 * avg_c))
            cv2.line(vis_img, (kp1[1], kp1[2]), (kp2[1], kp2[2]), SKELETON_COLOR, thickness)

    return vis_img


def main():
    args = parse_args()

    if args.model and (not args.opts or 'TEST.MODEL_FILE' not in args.opts):
        if args.opts is None:
            args.opts = []
        args.opts.extend(['TEST.MODEL_FILE', args.model])

    update_config(cfg, args)

    # cuDNN settings
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    print('=> Loading model from {}'.format(args.model))
    model_p = load_model(cfg, args.model)

    in_h, in_w = int(cfg.MODEL.IMAGE_SIZE[0]), int(cfg.MODEL.IMAGE_SIZE[1])

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    to_tensor = transforms.Compose([transforms.ToTensor(), normalize])

    src = args.input
    if src.isdigit():
        cap = cv2.VideoCapture(int(src))
    else:
        cap = cv2.VideoCapture(src)

    if not cap.isOpened():
        raise RuntimeError('Cannot open video source: {}'.format(args.input))

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps:  # NaN or 0 fallback
        fps = 25.0
    out_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    out_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (out_w, out_h))
    if not out.isOpened():
        cap.release()
        raise RuntimeError('Cannot open VideoWriter for {}'.format(args.output))

    print('=> Processing video: {} ({}x{} @ {} FPS)'.format(args.input, out_w, out_h, fps))
    print('=> Writing to: {}'.format(args.output))

    frame_idx = 0
    t0 = time.time()
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

        
            h, w = frame_bgr.shape[:2]
            center = np.array([w / 2.0, h / 2.0], dtype=np.float32)
            scale  = np.array([w / 200.0, h / 200.0], dtype=np.float32)
            trans  = get_affine_transform(center, scale, 0, (in_w, in_h))

            input_img = cv2.warpAffine(frame_bgr, trans, (in_w, in_h), flags=cv2.INTER_LINEAR)
            vis_img = input_img.copy() 

            inp = to_tensor(cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)).unsqueeze(0).cuda()

            with torch.no_grad():
                _, output = model_p(inp)

                dummy_center = np.array([in_w / 2.0, in_h / 2.0], dtype=np.float32)
                dummy_scale  = np.array([in_w / 200.0, in_h / 200.0], dtype=np.float32)
                preds, maxvals = get_final_preds(
                    cfg,
                    output.clone().detach().cpu().numpy(),
                    np.array([dummy_center]),
                    np.array([dummy_scale])
                )

            vis_img = draw_pose(vis_img, preds, maxvals, conf_thresh=args.conf_threshold)

            vis_out = cv2.resize(vis_img, (out_w, out_h), interpolation=cv2.INTER_AREA)

            dt = (time.time() - t0) / max(1, frame_idx + 1)
            fps_now = 1.0 / dt if dt > 0 else 0.0
            cv2.putText(vis_out, f'FiDIP Video Demo | {fps_now:.1f} FPS',
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 2, cv2.LINE_AA)

            out.write(vis_out)
            frame_idx += 1

    finally:
        cap.release()
        out.release()

    total_t = time.time() - t0
    print(f'=> Done. {frame_idx} frames written to {args.output} in {total_t:.2f}s '
          f'({(frame_idx/total_t if total_t>0 else 0):.2f} FPS avg).')


if __name__ == '__main__':
    main()
