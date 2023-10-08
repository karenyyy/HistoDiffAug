import os
import pickle
import random
import shutil
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import mutual_info_score
from torchvision.transforms import transforms
from umap.umap_ import UMAP
import matplotlib.pyplot as plt
from umap import plot

# from ldm.models.autoencoder import AutoencoderKL
from vision_transformer import vit_small


model = vit_small(
            patch_size=16,
            drop_rate=0.0,
            drop_path_rate=0.1,
            attn_drop_rate=0.0,
            use_mean_pooling=0,
            num_classes=9,
        )


state_dict = torch.load('/data/karenyyy/ssl_slide2/train5/checkpoint_teacher_cls.pth', map_location="cpu")
state_dict = state_dict['model']
state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}

msg = model.load_state_dict(state_dict, strict=False)

print('Pretrained weights found and loaded with msg: {}'.format(msg))

model.eval()

def normalize_tensor(x,eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x**2,dim=1,keepdim=True))
    return x/(norm_factor+eps)

def cal_latent_feature_dis(img1_path, img2_paths, pairs, mi_lst, dis_lst):
    # global autoencoder
    global model
    img1 = Image.open(img1_path)
    arr1 = np.array(img1).mean(1)

    img1 = transforms.ToTensor()(img1).unsqueeze(dim=0).to('cuda:7')

    model = model.to(img1.device)
    feature1 = normalize_tensor(model(img1)[0].detach())

    for img2_path in img2_paths:
        img2 = Image.open(img2_path)
        arr2 = np.array(img2).mean(1)

        img2 = transforms.ToTensor()(img2).unsqueeze(dim=0).to('cuda:7')

        feature2 = normalize_tensor(model(img2)[0].detach())

        pairs.append(f'{img1_path.split("/")[1]} - {img2_path.split("/")[1]}')
        mi_lst.append(mutual_info_score(arr1.ravel(), arr2.ravel()))

        dis_lst.append(((feature1 - feature2) ** 2).mean().item())

    return pairs, mi_lst, dis_lst

def summarize_dis_metrics(target_tissue):
    path = 'crc_examples'
    tissue_types = list(os.listdir(path))

    target_patches = os.listdir(os.path.join(path, target_tissue))

    for idx in range(100):
        target_file = target_patches[random.randint(1, len(target_patches)-1)]

        file_path = []
        for tissue_type in tissue_types:
            patches = os.listdir(os.path.join(path, tissue_type))
            file = patches[random.randint(1, len(patches)-1)]
            file_path.append(os.path.join(path, tissue_type, file))
        pairs, mi_lst, dis_lst = [], [], []
        pairs, mi_lst, dis_lst = cal_latent_feature_dis(img1_path=os.path.join(path, target_tissue, target_file),
                               img2_paths=file_path,
                               pairs=pairs, mi_lst=mi_lst, dis_lst=dis_lst)

        df = pd.DataFrame(
            {
                'pairs': pairs,
                'dis': dis_lst,
                'mi': mi_lst
            }
        )

        # df_grouped = df.groupby('pairs').agg({'dis':'mean'})
        df_sorted = df.sort_values(['dis', 'mi'], ascending=[True, False])

        print(idx, df_sorted)

import warnings
warnings.filterwarnings("ignore")

with open('saved_pkls/crc5_train5+fake100%_centroid_example.pkl', 'rb') as handle:
    centroid_dct = pickle.load(handle)
model = model.to('cuda:7')

CLASS2IDX = {
    'ADI': 0,
    'BACK': 1,
    'DEB': 2,
    'LYM': 3,
    'MUC': 4,
    'MUS': 5,
    'NORM': 6,
    'STR': 7,
    'TUM': 8
}

for c, c_idx in CLASS2IDX.items():
    file_name, dis_dct = [], defaultdict(list)
    for img2_path in os.listdir('/data/karenyyy/latent-diffusion2/fake_examples_crc5'):
        tissue_type = int(img2_path.split('_')[1])

        if tissue_type == c_idx:
        # if True:
            file_name.append(img2_path)
            img2 = Image.open(os.path.join('/data/karenyyy/latent-diffusion2/fake_examples_crc5', img2_path)).resize((224, 224))
            img2 = transforms.ToTensor()(img2).unsqueeze(dim=0).to('cuda:7')
            feature2 = normalize_tensor(model(img2)[0].detach())

            for k, v in centroid_dct.items():
                feature_cen = centroid_dct[k]
                dis_dct[k].append(
                        ((feature_cen - feature2.data.cpu().numpy()) ** 2).mean())

    df = pd.DataFrame(
        dis_dct
    )

    df['idxmin'] = df.idxmin(axis=1)
    df['file_name'] = file_name
    df_sorted = df.sort_values([c], ascending=[True])

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    # print(df_sorted)
    print(df_sorted[df_sorted['idxmin']==c], df_sorted[df_sorted['idxmin']==c].shape)
    # df_ori = pd.read_csv(f'saved_csvs/{c}.csv')
    # df_added = df_ori.append(df_sorted[df_sorted['idxmin']==c], ignore_index=True)
    # print(df_added)
    # df_added.to_csv(f'saved_csvs/{c}.csv', index=False)
    df_sorted[df_sorted['idxmin']==c].to_csv(f'saved_csvs_crc5_train5+fake100_centroid_example.pkl/{c}.csv', index=False)
    print(f'{c}.csv saved!')



import pandas as pd
save_path = '/data/karenyyy/CRC_Data/train5+crc5_fake300%'
for tissue_file in os.listdir('saved_csvs_crc5_train5+fake100_centroid_example.pkl'):
    if 'csv' in tissue_file:
        tissue_type = tissue_file.replace('.csv', '')
        df = pd.read_csv(os.path.join('saved_csvs_crc5_train5+fake100_centroid_example.pkl', tissue_file))
        # if tissue_type == 'STR':
        #     df = pd.read_csv(os.path.join('saved_csvs', tissue_file))

        # df = pd.read_csv(os.path.join('saved_gan_csvs', tissue_file))
        cnt = 0

        for idx, row in df.iterrows():
            if cnt <= 500:
                src_path = os.path.join('/data/karenyyy/latent-diffusion2/fake_examples_crc5', row['file_name'])

                try:
                    os.mkdir(os.path.join(save_path, row['idxmin']))
                except:
                    pass
                dst_path = os.path.join(os.path.join(save_path, row['idxmin']), row['file_name'])

                # dst_path = os.path.join(save_path, row['idxmin'] + '_' + row['file_name'])
                if not os.path.exists(dst_path):
                    shutil.copyfile(src=src_path, dst=dst_path)
                    print(f'{src_path} copied to {dst_path}')
                    cnt += 1
