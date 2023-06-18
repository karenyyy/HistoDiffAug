import os
import pickle
import random
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

crc_val = '/data/karenyyy/CRC_Data/train5+crc5_fake300%'
feature_dct = defaultdict(list)
feature_to_plot = []
labels_to_plot = []
for tissue_type in os.listdir(crc_val):
    for idx, patch in enumerate(os.listdir(os.path.join(crc_val, tissue_type))):
        print(tissue_type, idx)
        # if idx > 100:
        #     break
        img_path = os.path.join(crc_val, tissue_type, patch)
        img = Image.open(img_path)
        img = transforms.ToTensor()(img).unsqueeze(dim=0).to('cuda:7')
        model = model.to(img.device)
        feature = normalize_tensor(model(img)[0].detach())
        feature_dct[tissue_type].append(feature.data.cpu().numpy())

        # b, c, w, h = feature.shape
        feature_to_plot.append(feature.data.cpu().numpy())
        labels_to_plot.append(tissue_type)


centroid_dct = defaultdict(list)
for tissue, features in feature_dct.items():
    centroid = np.stack(features, axis=0).mean(axis=0, keepdims=False)
#     feature_to_plot.append(centroid.reshape(b, c * w * h))
#     labels_to_plot.append(tissue + '_c')
    centroid_dct[tissue] = centroid


with open("saved_pkls/crc5_train5+fake300%_centroid_example.pkl", 'wb') as handle:
    pickle.dump(centroid_dct, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('noft_train5+fake300%_centroid_example.pkl saved to saved_pkls')


embeddings = np.concatenate(feature_to_plot, axis=0)
mapper = UMAP(n_neighbors=10, min_dist=0.1).fit(embeddings)
plt.figure(figsize=(20, 10), dpi=300)
plot.points(mapper, cmap='Blues',
            # color_key={'0': 'green', '1': 'red'},
            labels=np.array(labels_to_plot),
            width=600, height=600,
            show_legend=False
            # background='black'
            )
# plt.savefig('figs/ae_crc9_umap_tmp.png')

plt.show()
