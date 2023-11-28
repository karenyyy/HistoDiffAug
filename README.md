# HistoDiffAug dev


## Download slides from TCGA

- Download manifests from each disease type: [TCGA Site](https://portal.gdc.cancer.gov/projects?filters=%7B%22op%22%3A%22and%22%2C%22content%22%3A%5B%7B%22op%22%3A%22in%22%2C%22content%22%3A%7B%22field%22%3A%22projects.program.name%22%2C%22value%22%3A%5B%22TCGA%22%5D%7D%7D%5D%7D)
- Install download software: [GDC Data Transfer Tool](https://gdc.cancer.gov/access-data/gdc-data-transfer-tool)
- Downloading slides using the manifest File of each disease type:
  ```python
  gdc-client download -m gdc_manifest.<cancer-type>.txt
  ```

## Tile slides from TCGA
```python
python wsi_tiling.py  \
  --dataset <path of dataset containing svs>  \
  --output <path to store tiled patches>  \
  --scale 20 --patch_size 1024 --num_threads 16
```
