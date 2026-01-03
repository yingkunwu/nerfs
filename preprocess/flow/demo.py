import sys
sys.path.append('core')

import argparse
import os
import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from utils.utils import InputPadder
from utils.frame_utils import writeFlow


DEVICE = 'cuda'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def demo(args):
    # Load model
    model = RAFT(args)
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model))
    model = model.module.to(DEVICE).eval()

    # Prepare output directories
    base_dir = os.path.dirname(args.path)
    flow_fw_dir = os.path.join(base_dir, 'flow_fw')
    flow_bw_dir = os.path.join(base_dir, 'flow_bw')
    os.makedirs(flow_fw_dir, exist_ok=True)
    os.makedirs(flow_bw_dir, exist_ok=True)

    print('Predicting optical flow ...')
    images = sorted(glob.glob(os.path.join(args.path, '*')))

    with torch.no_grad():
        for imfile1, imfile2 in zip(images[:-1], images[1:]):
            img1 = load_image(imfile1)
            img2 = load_image(imfile2)
            padder = InputPadder(img1.shape)
            img1, img2 = padder.pad(img1, img2)

            _, flow_fw = model(img1, img2, iters=20, test_mode=True)
            _, flow_bw = model(img2, img1, iters=20, test_mode=True)

            stem = os.path.splitext(os.path.basename(imfile1))[0]
            flow_fw_path = os.path.join(flow_fw_dir, f'{stem}.flo')
            flow_bw_path = os.path.join(flow_bw_dir, f'{stem}.flo')

            writeFlow(flow_fw_path, flow_fw[0].permute(1, 2, 0).cpu().numpy())
            writeFlow(flow_bw_path, flow_bw[0].permute(1, 2, 0).cpu().numpy())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    demo(args)
