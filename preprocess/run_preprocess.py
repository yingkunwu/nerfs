import os
import glob
import argparse
from tqdm import tqdm
from pathlib import Path
import cv2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, help='data root directory', required=True)
    parser.add_argument('--cuda-device', type=str, default='0', help='cuda device to use')
    parser.add_argument('--max-width', type=int, default=1280, help='max image width')
    parser.add_argument('--max-height', type=int, default=720, help='max image height')
    parser.add_argument(
        '--images-resized', default='images_resized', help='location for resized/renamed images'
    )
    parser.add_argument('--image_input', default='frames', help='location for original images')
    parser.add_argument(
        '--undistorted-output', default='images', help='location of undistorted images'
    )
    parser.add_argument(
        '--overwrite', default=False, action='store_true', help='overwrite cache'
    )
    args = parser.parse_args()
    return args


def resize_frames(args):
    vid_name = os.path.basename(args.root_dir)
    frames_dir = os.path.join(args.root_dir, args.images_resized)
    os.makedirs(frames_dir, exist_ok=True)

    files = sorted(
        glob.glob(os.path.join(args.root_dir, args.image_input, '*.jpg')) +
        glob.glob(os.path.join(args.root_dir, args.image_input, '*.png'))
    )

    print('Resizing images ...')
    for file_ind, file in enumerate(tqdm(files, desc=f'imresize: {vid_name}')):
        out_frame_fn = f'{frames_dir}/{file_ind:05}.png'

        if os.path.exists(out_frame_fn) and not args.overwrite:
            continue

        im = cv2.imread(file)

        if im.shape[1] > args.max_width or im.shape[0] > args.max_height:
            factor = max(im.shape[1] / args.max_width, im.shape[0] / args.max_height)
            dsize = (int(im.shape[1] / factor), int(im.shape[0] / factor))
            im = cv2.resize(src=im, dsize=dsize, interpolation=cv2.INTER_AREA)

        cv2.imwrite(out_frame_fn, im)


def generate_masks(args):
    undist_dir = os.path.join(args.root_dir, args.undistorted_output)
    if not os.path.exists(undist_dir) or args.overwrite:
        os.makedirs(undist_dir, exist_ok=True)
        os.system(
            f'cp -r {args.root_dir}/{args.images_resized}/*.png {args.root_dir}/images'
        )
        os.system(
            f'CUDA_VISIBLE_DEVICES={args.cuda_device} python preprocess/predict_mask.py '
            f'--root_dir {args.root_dir}'
        )
        os.system(f'rm -r {args.root_dir}/images')


def run_colmap(args):
    if not os.path.exists(f'{args.root_dir}/database.db') or args.overwrite:
        os.system(
            f'CUDA_VISIBLE_DEVICES={args.cuda_device} colmap feature_extractor '
            f'--database_path={args.root_dir}/database.db '
            f'--image_path={args.root_dir}/{args.images_resized} '
            f'--ImageReader.mask_path={args.root_dir}/masks '
            f'--ImageReader.camera_model=SIMPLE_RADIAL '
            f'--ImageReader.single_camera=1 '
            f'--ImageReader.default_focal_length_factor=0.95 '
            f'--SiftExtraction.peak_threshold=0.004 '
            f'--SiftExtraction.max_num_features=8192 '
            f'--SiftExtraction.edge_threshold=16'
        )

        os.system(
            f'CUDA_VISIBLE_DEVICES={args.cuda_device} colmap exhaustive_matcher '
            f'--database_path={args.root_dir}/database.db '
        )

    if not os.path.exists(f'{args.root_dir}/sparse') or args.overwrite:
        os.makedirs(os.path.join(args.root_dir, 'sparse'), exist_ok=True)
        os.system(
            f'CUDA_VISIBLE_DEVICES={args.cuda_device} colmap mapper '
            f'--database_path={args.root_dir}/database.db '
            f'--image_path={args.root_dir}/{args.images_resized} '
            f'--output_path={args.root_dir}/sparse'
        )

    undist_dir = os.path.join(args.root_dir, args.undistorted_output)
    if not os.path.exists(undist_dir) or args.overwrite:
        os.makedirs(undist_dir, exist_ok=True)
        os.system(
            f'CUDA_VISIBLE_DEVICES={args.cuda_device} colmap image_undistorter '
            f'--input_path={args.root_dir}/sparse/0 '
            f'--image_path={args.root_dir}/{args.images_resized} '
            f'--output_path={args.root_dir} '
            f'--output_type=COLMAP'
        )


def generate_depth(args):
    disp_dir = os.path.join(args.root_dir, 'disps')
    if not os.path.exists(disp_dir) or args.overwrite:
        cur_dir = Path(__file__).absolute().parent
        os.chdir(f'{str(cur_dir)}/depth')
        os.environ['MKL_THREADING_LAYER'] = 'GNU'
        os.system(
            f'CUDA_VISIBLE_DEVICES={args.cuda_device} python run_monodepth.py '
            f'-i {cur_dir}/../{args.root_dir}/{args.images_resized} '
            f'-o {cur_dir}/../{args.root_dir}/disps -t dpt_large '
        )
        os.chdir(str(cur_dir))


def generate_flow(args):
    flow_fw_dir = os.path.join(args.root_dir, 'flow_fw')
    flow_bw_dir = os.path.join(args.root_dir, 'flow_bw')
    if not os.path.exists(flow_fw_dir) or not os.path.exists(flow_bw_dir) or args.overwrite:
        cur_dir = Path(__file__).absolute().parent
        os.chdir(f'{str(cur_dir)}/flow')
        os.system(
            f'CUDA_VISIBLE_DEVICES={args.cuda_device} python demo.py '
            f'--model raft-things.pth '
            f'--path {cur_dir}/../{args.root_dir}/{args.images_resized}'
        )
        os.chdir(str(cur_dir))


def main():
    args = parse_args()
    resize_frames(args)
    generate_masks(args)
    run_colmap(args)
    generate_depth(args)
    generate_flow(args)
    print('finished!')


if __name__ == '__main__':
    main()
