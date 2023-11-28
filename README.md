# HistoDiffAug dev


## Download slides from TCGA

- Download manifests from each disease type: [tcga site](https://portal.gdc.cancer.gov/projects?filters=%7B%22op%22%3A%22and%22%2C%22content%22%3A%5B%7B%22op%22%3A%22in%22%2C%22content%22%3A%7B%22field%22%3A%22projects.program.name%22%2C%22value%22%3A%5B%22TCGA%22%5D%7D%7D%5D%7D)

```python
python wsi_tiling.py  \
  --dataset <path of dataset containing svs>  \
  --output <path to store tiled patches>  \
  --scale 20 --patch_size 1024 --num_threads 16
```
