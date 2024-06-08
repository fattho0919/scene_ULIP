'''
 * Copyright (c) 2023, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Le Xue
'''

import random

import torch
import numpy as np
import torch.utils.data as data

import yaml
from easydict import EasyDict

from utils.io import IO
from utils.build import DATASETS
from utils.logger import *
from utils.build import build_dataset_from_cfg
import json
from tqdm import tqdm
import pickle
from PIL import Image
from scipy.spatial import cKDTree

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

# def farthest_point_sample(point, npoint):
#     """
#     Input:
#         xyz: pointcloud data, [N, D]
#         npoint: number of samples
#     Return:
#         centroids: sampled pointcloud index, [npoint, D]
#     """
#     N, D = point.shape
#     xyz = point[:,:3]
#     centroids = np.zeros((npoint,))
#     distance = np.ones((N,)) * 1e10
#     farthest = np.random.randint(0, N)
#     for i in range(npoint):
#         centroids[i] = farthest
#         centroid = xyz[farthest, :]
#         dist = np.sum((xyz - centroid) ** 2, -1)
#         mask = dist < distance
#         distance[mask] = dist[mask]
#         farthest = np.argmax(distance, -1)
#     point = point[centroids.astype(np.int32)]
#     return point

def farthest_point_sampling(point_cloud, num_centers):
    farthest_pts = [np.random.randint(len(point_cloud))]
    distances = np.linalg.norm(point_cloud[:, :3] - point_cloud[farthest_pts[-1], :3], axis=1)
    for _ in range(num_centers - 1):
        next_pt = np.argmax(distances)
        farthest_pts.append(next_pt)
        distances = np.minimum(distances, np.linalg.norm(point_cloud[:, :3] - point_cloud[next_pt, :3], axis=1))
    return point_cloud[farthest_pts], farthest_pts

def k_nearest_neighbors(points, centers, k):
    tree = cKDTree(points[:, :3])
    neighbors = []
    for center in centers:
        _, idx = tree.query(center[:3], k=k)
        neighbors.append(points[idx] - center)
    return np.concatenate(neighbors, axis=0)

def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds with color information to augment the dataset.
        Rotation is per shape based along the up direction, without altering the color.
        Input:
          BxNx6 array, original batch of point clouds (x, y, z, r, g, b)
        Return:
          BxNx6 array, rotated batch of point clouds with unchanged color
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        # Select only x, y, z for rotation
        shape_pc = batch_data[k, :, :3]
        rotated_pc = np.dot(shape_pc, rotation_matrix)
        # Combine the rotated coordinates with the original colors
        rotated_data[k, :, :3] = rotated_pc
        rotated_data[k, :, 3:] = batch_data[k, :, 3:]  # Keep RGB colors unchanged
    return rotated_data

def random_point_dropout(batch_pc, max_dropout_ratio=0.875):
    ''' Applies random dropout to point clouds including color data.
        batch_pc: BxNx6, where N is the number of points per batch, and 6 represents x, y, z, r, g, b
    '''
    for b in range(batch_pc.shape[0]):
        dropout_ratio = np.random.random() * max_dropout_ratio  # Random dropout ratio between 0 and max_dropout_ratio
        drop_idx = np.where(np.random.random((batch_pc.shape[1])) <= dropout_ratio)[0]
        if len(drop_idx) > 0:
            batch_pc[b, drop_idx, :] = batch_pc[b, 0, :]  # Set dropped points to the attributes of the first point
    return batch_pc

def random_scale_point_cloud(batch_data, scale_low=0.8, scale_high=1.25):
    """ Randomly scale the point cloud in the xyz dimensions only. The rgb data remains unchanged.
        Input:
            BxNx6 array, original batch of point clouds with xyz and rgb data
        Return:
            BxNx6 array, scaled batch of point clouds with unchanged rgb data
    """
    B, N, _ = batch_data.shape
    scales = np.random.uniform(scale_low, scale_high, B)
    for batch_index in range(B):
        batch_data[batch_index, :, :3] *= scales[batch_index]  # Scale only x, y, z
        # Color data at indices 3, 4, 5 remains unchanged
    return batch_data

def shift_point_cloud(batch_data, shift_range=0.1):
    """ Randomly shift point cloud. Shift is per point cloud in xyz dimensions only.
        Input:
          BxNx6 array, original batch of point clouds with xyz and rgb data
        Return:
          BxNx6 array, shifted batch of point clouds with unchanged rgb data
    """
    B, N, _ = batch_data.shape
    shifts = np.random.uniform(-shift_range, shift_range, (B, 3))
    for batch_index in range(B):
        batch_data[batch_index, :, :3] += shifts[batch_index, :]  # Only shift x, y, z
    return batch_data

def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points in the xyz dimensions only. The rgb data remains unchanged.
        Input:
          BxNx6 array, original batch of point clouds with xyz and rgb data
        Return:
          BxNx6 array, jittered batch of point clouds with unchanged rgb data
    """
    B, N, _ = batch_data.shape
    assert(clip > 0)
    jittered_data = np.zeros_like(batch_data)
    jittered_data[:, :, :3] = np.clip(sigma * np.random.randn(B, N, 3), -1*clip, clip)  # Jitter only x, y, z
    jittered_data[:, :, :3] += batch_data[:, :, :3]
    jittered_data[:, :, 3:] = batch_data[:, :, 3:]  # Keep rgb data unchanged
    return jittered_data

def rotate_perturbation_point_cloud(batch_data, angle_sigma=0.06, angle_clip=0.18):
    """ Randomly perturb the point clouds by small rotations in xyz dimensions only.
        Input:
          BxNx6 array, original batch of point clouds with xyz and rgb data
        Return:
          BxNx6 array, rotated batch of point clouds with unchanged rgb data
    """
    rotated_data = np.zeros_like(batch_data)
    for k in range(batch_data.shape[0]):
        angles = np.clip(angle_sigma * np.random.randn(3), -angle_clip, angle_clip)
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(angles[0]), -np.sin(angles[0])],
                       [0, np.sin(angles[0]), np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                       [0, 1, 0],
                       [-np.sin(angles[1]), 0, np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                       [np.sin(angles[2]), np.cos(angles[2]), 0],
                       [0, 0, 1]])
        R = np.dot(Rz, np.dot(Ry, Rx))
        shape_pc = batch_data[k, :, :3]
        rotated_data[k, :, :3] = np.dot(shape_pc.reshape((-1, 3)), R)
        rotated_data[k, :, 3:] = batch_data[k, :, 3:]  # Keep rgb data unchanged
    return rotated_data

import os, sys, h5py

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

@DATASETS.register_module()
class s3dis(data.Dataset):
    def __init__(self, config):
        self.level = config.get('level')
        self.data_root = config.DATA_PATH
        self.mapping_path = os.path.join(self.data_root, f"s3dis_{self.level}_vit-gpt2_matching_idx")
        self.pc_path = config.PC_PATH
        self.image_path = config.IMAGE_PATH
        self.tokenizer = config.tokenizer
        self.clip_preprocessor = config.clip_preprocessor
        self.rendered_image_addr = config.IMAGE_PATH
        self.data_list_file = os.path.join(self.data_root, f's3dis_{self.level}_list.json')
        print_log(f'[DATASET] Loading s3dis_{self.level} dataset', logger='S3DIS')

        print_log(f'[DATASET] Open file {self.data_list_file}', logger='S3DIS')
        with open(self.data_list_file, 'r') as f:
            self.annos = json.load(f)
            print_log(f'[DATASET] {len(self.annos)} instances were loaded', logger='S3DIS')

        # *---- if the mapping dict exists, load it; otherwise, construct it ----*
        if self.level != 'scene':
            if os.path.exists(os.path.join(self.data_root, f'{self.level}_mapping_dict.pkl')):
                print_log(f'[DATASET] Loading {self.level} mapping dict', logger='S3DIS')
                with open(f'{self.data_root}/{self.level}_mapping_dict.pkl', 'rb') as pickle_file:
                    self.mapping_dict = pickle.load(pickle_file)
            else:
                print_log(f'[DATASET] Constructing {self.level} mapping dict', logger='S3DIS')
                self.mapping_dict = {}
                for anno in self.annos:
                    if anno['scene_id'] not in self.mapping_dict:
                        mapping_file = os.path.join(self.mapping_path, f"{anno['scene_id']}.pickle")
                        with open(mapping_file, 'rb') as f:
                            mapping_idx = pickle.load(f)
                        for k,v in mapping_idx.items():
                            self.mapping_dict[k] = v
                with open(f'{self.data_root}/{self.level}_mapping_dict.pkl', 'wb') as pickle_file:
                    pickle.dump(self.mapping_dict, pickle_file)

        self.uniform = True
        self.augment = False
        self.use_caption_templates = False
        # =================================================
        # TODO: disable for backbones except for PointNEXT!!!
        self.use_height = config.use_height
        # =================================================

        if self.augment:
            print_log(f'[DATASET] Using augmented point clouds', logger='S3DIS')

    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc

    def random_sample(self, pc, num):
        np.random.shuffle(self.permutation)
        pc = pc[self.permutation[:num]]
        return pc

    def __getitem__(self, idx):
        sample = self.annos[idx]

        pc = IO.get(os.path.join(self.pc_path, f"{sample['scene_id']}.npy")).astype(np.float16)
        if self.level != 'scene':
            try:
                pc = pc[self.mapping_dict[sample['id']]]
            except Exception as e:
                print(e)
                pc = pc[:len(self.mapping_dict[sample['id']])]

        if self.augment:
            pc = random_point_dropout(pc[None, ...])
            pc = random_scale_point_cloud(pc)
            pc = shift_point_cloud(pc)
            pc = rotate_perturbation_point_cloud(pc)
            pc = rotate_point_cloud(pc)
            pc = pc.squeeze()

        # 這邊gravity_dim是1代表它假設y軸是重力方向，所以會把y軸的值減掉最小值，這樣就會讓y軸的值變成正值，也就是高度
        # s3dis和scannet都是z軸是重力方向，所以這邊要改成2
        if self.use_height:
            print_log(f'[DATASET] Using height', logger='S3DIS')
            self.gravity_dim = 2
            height_array = pc[:, self.gravity_dim:self.gravity_dim + 1] - pc[:,
                                                                       self.gravity_dim:self.gravity_dim + 1].min()
            pc = np.concatenate((pc, height_array), axis=1)
            pc = torch.from_numpy(pc).to(torch.bfloat16)
        else:
            pc = torch.from_numpy(pc).to(torch.bfloat16)

        captions = sample['conversations'][1]['value']
        # captions = [caption.strip() for caption in captions.split(',') if caption.strip()]
        # caption = random.choice(captions)
        # captions = []
        tokenized_captions = []
        if self.use_caption_templates:
            print("use caption templates")
            for template in self.templates:
                caption = template.format(caption)
                captions.append(caption)
                tokenized_captions.append(self.tokenizer(caption))
        else:
            # tokenized_captions.append(self.tokenizer(caption))
            tokenized_captions = self.tokenizer(captions)

        # tokenized_captions = torch.stack(tokenized_captions)
        # print(tokenized_captions.size())

        if self.level == 'view':
            image_dir = os.path.join(self.image_path, f"{sample['id']}.png")
            try:
                image = pil_loader(image_dir)
                image = self.clip_preprocessor(image)
            except:
                raise ValueError("image is corrupted: {}".format(image_dir))

            return tokenized_captions, pc, image
        else:
            return tokenized_captions, pc

    def __len__(self):
        return len(self.annos)
    
@DATASETS.register_module()
class scannet(data.Dataset):
    def __init__(self, config):
        self.level = config.get('level')
        self.data_root = config.DATA_PATH
        self.mapping_path = os.path.join(self.data_root, f"scannetv2_{self.level}_vit-gpt2_matching_idx.pickle")
        self.pc_path = config.PC_PATH
        self.image_path = config.IMAGE_PATH
        self.tokenizer = config.tokenizer
        self.clip_preprocessor = config.clip_preprocessor
        self.rendered_image_addr = config.IMAGE_PATH
        self.data_list_file = os.path.join(self.data_root, f'scannet_{self.level}_list.json')
        print_log(f'[DATASET] Loading scannet_{self.level} dataset', logger='SCANNET')

        print_log(f'[DATASET] Open file {self.data_list_file}', logger='SCANNET')
        with open(self.data_list_file, 'r') as f:
            self.annos = json.load(f)
            print_log(f'[DATASET] {len(self.annos)} instances were loaded', logger='SCANNET')

        # *---- if the mapping dict exists, load it; otherwise, construct it ----*
        if self.level != 'scene':
            print_log(f'[DATASET] Loading {self.level} mapping dict', logger='SCANNET')
            with open(self.mapping_path, 'rb') as pickle_file:
                self.mapping_dict = pickle.load(pickle_file)

        self.uniform = True
        self.augment = False
        self.use_caption_templates = False
        # =================================================
        # TODO: disable for backbones except for PointNEXT!!!
        self.use_height = config.use_height
        # =================================================

        if self.augment:
            print_log(f'[DATASET] Using augmented point clouds', logger='SCANNET')

    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc

    def random_sample(self, pc, num):
        np.random.shuffle(self.permutation)
        pc = pc[self.permutation[:num]]
        return pc

    def __getitem__(self, idx):
        i = 0
        sample = self.annos[idx]
        captions = sample['conversations'][1]['value']

        if self.level == 'entity':
            while (captions == '') or (sample['scene_id'] not in self.mapping_dict) or (sample['id'] not in self.mapping_dict[sample['scene_id']]):
                # print_log(f'Skip {sample["scene_id"]} {sample["id"]}', logger='DATASET')
                sample = self.annos[idx+i]
                captions = sample['conversations'][1]['value']
                i += 1
        elif self.level == 'view':
            while (captions == '') or (sample['scene_id'] not in self.mapping_dict) or (sample['id'] not in self.mapping_dict[sample['scene_id']]) or (os.path.exists(os.path.join(self.image_path, f"{sample['scene_id']}",f"{sample['id']}.jpg")) == False):
                # print_log(f'Skip {sample["scene_id"]} {sample["id"]}', logger='DATASET')
                sample = self.annos[idx+i]
                captions = sample['conversations'][1]['value']
                i += 1
        # captions = [caption.strip() for caption in captions.split(',') if caption.strip()]
        # caption = random.choice(captions)
        # captions = []
        tokenized_captions = []
        if self.use_caption_templates:
            print("use caption templates")
            for template in self.templates:
                caption = template.format(caption)
                captions.append(caption)
                tokenized_captions.append(self.tokenizer(caption))
        else:
            # tokenized_captions.append(self.tokenizer(caption))
            tokenized_captions = self.tokenizer(captions)

        pc = IO.get(os.path.join(self.pc_path, f"{sample['scene_id']}.npy")).astype(np.float16)
        if self.level != 'scene':
            try:
                pc = pc[self.mapping_dict[sample['scene_id']][sample['id']]]
            except Exception as e:
                print(e)
                pc = pc[:len(self.mapping_dict[sample['scene_id']][sample['id']])]
                

        if self.augment:
            pc = random_point_dropout(pc[None, ...])
            pc = random_scale_point_cloud(pc)
            pc = shift_point_cloud(pc)
            pc = rotate_perturbation_point_cloud(pc)
            pc = rotate_point_cloud(pc)
            pc = pc.squeeze()

        # 這邊gravity_dim是1代表它假設y軸是重力方向，所以會把y軸的值減掉最小值，這樣就會讓y軸的值變成正值，也就是高度
        # s3dis和scannet都是z軸是重力方向，所以這邊要改成2
        if self.use_height:
            print_log(f'[DATASET] Using height', logger='SCANNET')
            self.gravity_dim = 2
            height_array = pc[:, self.gravity_dim:self.gravity_dim + 1] - pc[:,
                                                                       self.gravity_dim:self.gravity_dim + 1].min()
            pc = np.concatenate((pc, height_array), axis=1)
            pc = torch.from_numpy(pc).to(torch.bfloat16)
        else:
            pc = torch.from_numpy(pc).to(torch.bfloat16)

        # tokenized_captions = torch.stack(tokenized_captions)
        # print(tokenized_captions.size())

        if self.level == 'view':
            image_dir = os.path.join(self.image_path, f"{sample['scene_id']}",f"{sample['id']}.jpg")
            try:
                image = pil_loader(image_dir)
                image = self.clip_preprocessor(image)
            except:
                raise ValueError("image is corrupted: {}".format(image_dir))

            return tokenized_captions, pc, image
        else:
            return tokenized_captions, pc

    def __len__(self):
        return len(self.annos)

import collections.abc as container_abcs
int_classes = int
from torch._six import string_classes

import re
default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")
np_str_obj_array_pattern = re.compile(r'[SaUO]')

def view_customized_collate_fn(batch):
    tokenized_captions, pcs, images = zip(*batch)
    
    tokenized_captions = torch.cat(tokenized_captions, dim=0)
    images = torch.stack(images)

    min_num_points = min(pc.shape[0] for pc in pcs)
    num_centers = 256  # Number of center points to sample
    k = min_num_points // num_centers  # Number of neighbors per center point

    processed_pcs = []
    for pc in pcs:
        centers, _ = farthest_point_sampling(pc, num_centers)
        processed_pc = k_nearest_neighbors(pc, centers, k)
        processed_pcs.append(torch.from_numpy(processed_pc).float())

    pcs = torch.stack(processed_pcs)

    return tokenized_captions, pcs, images

def customized_collate_fn(batch):
    tokenized_captions, pcs = zip(*batch)
    
    tokenized_captions = torch.cat(tokenized_captions, dim=0)

    min_num_points = min(pc.shape[0] for pc in pcs)
    num_centers = 32  # Number of center points to sample
    k = min_num_points // num_centers  # Number of neighbors per center point

    processed_pcs = []
    for pc in pcs:
        centers, _ = farthest_point_sampling(pc, num_centers)
        processed_pc = k_nearest_neighbors(pc, centers, k)
        processed_pcs.append(torch.from_numpy(processed_pc).float())

    pcs = torch.stack(processed_pcs)

    return tokenized_captions, pcs


def merge_new_config(config, new_config):
    for key, val in new_config.items():
        if not isinstance(val, dict):
            if key == '_base_':
                with open(new_config['_base_'], 'r') as f:
                    try:
                        val = yaml.load(f, Loader=yaml.FullLoader)
                    except:
                        val = yaml.load(f)
                config[key] = EasyDict()
                merge_new_config(config[key], val)
            else:
                config[key] = val
                continue
        if key not in config:
            config[key] = EasyDict()
        merge_new_config(config[key], val)
    return config

def cfg_from_yaml_file(cfg_file):
    config = EasyDict()
    with open(cfg_file, 'r') as f:
        new_config = yaml.load(f, Loader=yaml.FullLoader)
    merge_new_config(config=config, new_config=new_config)
    return config

class Dataset_3D():
    def __init__(self, args, tokenizer, dataset_name, clip_preprocessor=None, level="scene"):
        self.dataset_name = dataset_name
        # if dataset_type == 'train':
        #     self.dataset_name = args.pretrain_dataset_name
        # elif dataset_type == 'val':
        #     self.dataset_name = args.validate_dataset_name
        # else:
        #     raise ValueError("not supported dataset type.")
        with open('./data/dataset_catalog.json', 'r') as f:
            self.dataset_catalog = json.load(f)
            self.dataset_usage = self.dataset_catalog[self.dataset_name]['usage'] # train or val
            self.dataset_split = self.dataset_catalog[self.dataset_name][self.dataset_usage] # dataset_catalog[dataset_name]["train" / "val"]
            self.dataset_config_dir = self.dataset_catalog[self.dataset_name]['config']
        self.tokenizer = tokenizer
        self.clip_preprocessor = clip_preprocessor
        self.pretrain_dataset_prompt = self.dataset_name
        # self.validate_dataset_prompt = args.validate_dataset_prompt
        self.build_3d_dataset(args, self.dataset_config_dir, level)

    def build_3d_dataset(self, args, config, level):
        config = cfg_from_yaml_file(config)
        config.tokenizer = self.tokenizer
        config.clip_preprocessor = self.clip_preprocessor
        config.pretrain_dataset_prompt = self.dataset_name
        # config.validate_dataset_prompt = self.validate_dataset_prompt
        config.args = args
        config.use_height = args.use_height
        config.npoints = args.npoints
        config_others = EasyDict({'subset': self.dataset_split, 'whole': True, 'level': level})
        self.dataset = build_dataset_from_cfg(config, config_others)
