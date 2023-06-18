### To be completed and organized, 70% done, need to fix some hard-coding in scripts


# Title of Paper

This repository contains the code used in the MICCAI 23' paper titled "Title of Paper".

## Abstract

Provide a brief summary of the research paper.

## Package Requirements

  - python=3.8.5
  - pip=20.3
  - cudatoolkit=11.0
  - pytorch=1.7.0
  - torchvision=0.8.1
  - numpy=1.19.2
  - albumentations==0.4.3
  - opencv-python==4.1.2.30
  - pudb==2019.2
  - imageio==2.9.0
  - imageio-ffmpeg==0.4.2
  - pytorch-lightning==1.4.2
  - omegaconf==2.1.1
  - test-tube>=0.7.5
  - streamlit>=0.73.1
  - einops==0.3.0
  - torch-fidelity==0.3.0
  - transformers==4.3.1
  - taming-transformers
  - clip

## Usage

### autoencoder training

```python
python main.py --base configs/autoencoder/autoencoder_histo_kl_64x64x3.yaml -t --gpus 0,1,2,3
```


### diffusion training 

```python
python main.py --base configs/latent-diffusion/histo-ldm-kl-8.yaml -t --gpus 0,1,2,3
```


### classifier training

```python
CUDA_VISIBLE_DEVICES=0 python classifier_train.py --data_dir /data/karenyyy/PCam/train10+fake \
                                    --val_data_dir /data/karenyyy/PCam/val10 \
                                    --iterations 300000 \
                                    --anneal_lr True \
                                    --batch_size 32 \
                                    --lr 5e-5 \
                                    --save_interval 100 \
                                    --weight_decay 0.05 \
                                    --image_size 64 \
                                    --classifier_attention_resolutions 32,16 \
                                    --classifier_depth 2 \
                                    --classifier_width 128 \
                                     --classifier_pool attention \
                                    --classifier_resblock_updown True \
                                    --classifier_use_scale_shift_norm True
```


### DDIM sampling

```python
python sample_diffusion.py -r /data/histo_diffusion_augmentation/diffusion_model.ckpt -n 10 -c 10
```

#### centroid-based filtering

```python
python distance.py
```



```python

### evaluation 

#### crc 9 class classification

```python
CUDA_VISIBLE_DEVICES=0 python eval.py \
                                --arch 'vit_small' \
                                --data_path /data/karenyyy/CRC_Data \
                                --output_dir train5+noft_randomfake50%
```

#### umap visualization 

```python
python vis.py
```


## Citation

If you use this code in your own research, please cite the following paper:
... [Insert citation here]


## License

Specify the license for the code, if applicable.

## Contact

For any questions or inquiries, please send email to ...
