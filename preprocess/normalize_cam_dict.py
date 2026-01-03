import os
import numpy as np
import json
import copy
import open3d as o3d
import argparse
from pyquaternion import Quaternion
import trimesh
import random
import shutil

from read_write_model import read_model


def parse_tracks(colmap_images, colmap_points3D):
    all_tracks = []     # list of dicts; each dict represents a track
    all_points = []     # list of all 3D points
    view_keypoints = {} # dict of lists; each list represents the triangulated key points of a view


    for point3D_id in colmap_points3D:
        point3D = colmap_points3D[point3D_id]
        image_ids = point3D.image_ids
        point2D_idxs = point3D.point2D_idxs

        cur_track = {}
        cur_track['xyz'] = (point3D.xyz[0], point3D.xyz[1], point3D.xyz[2])
        cur_track['err'] = point3D.error.item()

        cur_track_len = len(image_ids)
        assert (cur_track_len == len(point2D_idxs))
        all_points.append(list(cur_track['xyz'] + (cur_track['err'], cur_track_len) + tuple(point3D.rgb)))

        pixels = []
        for i in range(cur_track_len):
            image = colmap_images[image_ids[i]]
            img_name = image.name
            point2D_idx = point2D_idxs[i]
            point2D = image.xys[point2D_idx]
            assert (image.point3D_ids[point2D_idx] == point3D_id)
            pixels.append((img_name, point2D[0], point2D[1]))

            if img_name not in view_keypoints:
                view_keypoints[img_name] = [(point2D[0], point2D[1]) + cur_track['xyz'] + (cur_track_len, ), ]
            else:
                view_keypoints[img_name].append((point2D[0], point2D[1]) + cur_track['xyz'] + (cur_track_len, ))

        cur_track['pixels'] = sorted(pixels, key=lambda x: x[0]) # sort pixels by the img_name
        all_tracks.append(cur_track)

    return all_tracks, all_points, view_keypoints


def parse_camera_dict(colmap_cameras, colmap_images):
    camera_dict = {}
    for image_id in colmap_images:
        image = colmap_images[image_id]

        img_name = image.name
        cam = colmap_cameras[image.camera_id]

        # print(cam)
        assert(cam.model == 'PINHOLE')

        img_size = [cam.width, cam.height]
        params = list(cam.params)
        qvec = list(image.qvec)
        tvec = list(image.tvec)

        # w, h, fx, fy, cx, cy, qvec, tvec
        # camera_dict[img_name] = img_size + params + qvec + tvec
        camera_dict[img_name] = {}
        camera_dict[img_name]['img_size'] = img_size

        fx, fy, cx, cy = params
        K = np.eye(4)
        K[0, 0] = fx
        K[1, 1] = fy
        K[0, 2] = cx
        K[1, 2] = cy
        camera_dict[img_name]['K'] = list(K.flatten())

        rot = Quaternion(qvec[0], qvec[1], qvec[2], qvec[3]).rotation_matrix
        W2C = np.eye(4)
        W2C[:3, :3] = rot
        W2C[:3, 3] = np.array(tvec)
        camera_dict[img_name]['W2C'] = list(W2C.flatten())

    return camera_dict


def extract_all_to_dir(sparse_dir, out_dir, ext='.bin'):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    camera_dict_file = os.path.join(out_dir, 'kai_cameras.json')
    xyz_file = os.path.join(out_dir, 'kai_points.txt')
    track_file = os.path.join(out_dir, 'kai_tracks.json')
    keypoints_file = os.path.join(out_dir, 'kai_keypoints.json')
    
    colmap_cameras, colmap_images, colmap_points3D = read_model(sparse_dir, ext)

    camera_dict = parse_camera_dict(colmap_cameras, colmap_images)
    with open(camera_dict_file, 'w') as fp:
        json.dump(camera_dict, fp, indent=2, sort_keys=True)

    all_tracks, all_points, view_keypoints = parse_tracks(colmap_images, colmap_points3D)
    all_points = np.array(all_points)
    np.savetxt(xyz_file, all_points, header='# format: x, y, z, reproj_err, track_len, color(RGB)', fmt='%.6f')

    mesh = trimesh.Trimesh(vertices=all_points[:, :3].astype(np.float32), 
                           vertex_colors=all_points[:, -3:].astype(np.uint8))
    mesh.export(os.path.join(out_dir, 'kai_points.ply'))

    with open(track_file, 'w') as fp:
        json.dump(all_tracks, fp)

    with open(keypoints_file, 'w') as fp:
        json.dump(view_keypoints, fp)


def get_tf_cams(cam_dict, target_radius=1.):
    cam_centers = []
    for im_name in cam_dict:
        W2C = np.array(cam_dict[im_name]['W2C']).reshape((4, 4))
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center
    scale = target_radius / radius

    return translate, scale


def normalize_cam_dict(in_cam_dict_file, out_cam_dict_file, target_radius=1., in_geometry_file=None, out_geometry_file=None):
    with open(in_cam_dict_file) as fp:
        in_cam_dict = json.load(fp)

    translate, scale = get_tf_cams(in_cam_dict, target_radius=target_radius)

    if in_geometry_file is not None and out_geometry_file is not None:
        # check this page if you encounter issue in file io: http://www.open3d.org/docs/0.9.0/tutorial/Basic/file_io.html
        geometry = o3d.io.read_triangle_mesh(in_geometry_file)
        
        tf_translate = np.eye(4)
        tf_translate[:3, 3:4] = translate
        tf_scale = np.eye(4)
        tf_scale[:3, :3] *= scale
        tf = np.matmul(tf_scale, tf_translate)

        geometry_norm = geometry.transform(tf)
        o3d.io.write_triangle_mesh(out_geometry_file, geometry_norm)
  
    def transform_pose(W2C, translate, scale):
        C2W = np.linalg.inv(W2C)
        cam_center = C2W[:3, 3]
        cam_center = (cam_center + translate) * scale
        C2W[:3, 3] = cam_center
        return np.linalg.inv(C2W)

    out_cam_dict = copy.deepcopy(in_cam_dict)
    for img_name in out_cam_dict:
        W2C = np.array(out_cam_dict[img_name]['W2C']).reshape((4, 4))
        W2C = transform_pose(W2C, translate, scale)
        assert(np.isclose(np.linalg.det(W2C[:3, :3]), 1.))
        out_cam_dict[img_name]['W2C'] = list(W2C.flatten())

    with open(out_cam_dict_file, 'w') as fp:
        json.dump(out_cam_dict, fp, indent=2, sort_keys=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir',
                        type=str,
                        help='data root directory',
                        required=True)
    args = parser.parse_args()

    os.makedirs(os.path.join(args.root_dir, 'posed_images'), exist_ok=True)
    extract_all_to_dir(os.path.join(args.root_dir, 'sparse'),
                       os.path.join(args.root_dir, 'posed_images'))
    undistorted_img_dir = os.path.join(args.root_dir, 'images')
    posed_img_dir_link = os.path.join(args.root_dir, 'posed_images', 'images')
    if os.path.exists(posed_img_dir_link):
        os.remove(posed_img_dir_link)
    os.symlink(undistorted_img_dir, posed_img_dir_link)

    # normalize average camera center to origin, and put all cameras inside
    # the unit sphere
    in_cam_dict_file = os.path.join(
        args.root_dir, 'posed_images', 'kai_cameras.json')
    out_cam_dict_file = os.path.join(
        args.root_dir, 'posed_images', 'kai_cameras_normalized.json')
    normalize_cam_dict(in_cam_dict_file, out_cam_dict_file, target_radius=1.)

    # create train/validation split for nerf training

    # ratio of train/validation split
    train_ratio = 0.8  # 80% train, 20% validation
    save_dir = os.path.join(args.root_dir, 'split_data')

    # ==== Prepare output directories ====
    for subset in ["train", "val"]:
        os.makedirs(os.path.join(save_dir, subset, "pose"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, subset, "intrinsics"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, subset, "images"), exist_ok=True)

    # ==== Load JSON ====
    with open(out_cam_dict_file, "r") as f:
        data = json.load(f)

    keys = sorted(data.keys())  # ensures consistent order like 000001, 000002, ...
    seed = 42
    random.seed(seed)
    random.shuffle(keys)

    # determine split index
    split_idx = int(len(keys) * train_ratio)

    train_keys = keys[:split_idx]
    val_keys = keys[split_idx:]

    def save_matrix(matrix, path):
        arr = np.array(matrix)
        np.savetxt(path, arr, fmt="%.10f")

    # ==== Save files and copy images ====
    for subset, key_list in zip(["train", "val"], [train_keys, val_keys]):
        for key in key_list:
            base_name = os.path.splitext(key)[0]
            entry = data[key]

            # intrinsic and pose
            K = entry["K"]
            W2C = entry["W2C"]

            intrinsic_path = os.path.join(save_dir, subset, "intrinsics", f"{base_name}.txt")
            pose_path = os.path.join(save_dir, subset, "pose", f"{base_name}.txt")

            save_matrix(K, intrinsic_path)
            save_matrix(W2C, pose_path)

            # copy image from img_root to the appropriate subset images folder
            src_img = os.path.join(undistorted_img_dir, key)
            dst_img = os.path.join(save_dir, subset, "images", key)
            try:
                shutil.copy2(src_img, dst_img)
            except FileNotFoundError:
                print(f"⚠️ Image not found, skipping: {src_img}")

    print(f"✅ Done! {len(train_keys)} train and {len(val_keys)} validation samples saved.")
