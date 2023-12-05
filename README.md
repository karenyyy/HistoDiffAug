# HistoDiffAug dev


## Download slides from TCGA

- Download manifests from each disease type: [TCGA Site](https://portal.gdc.cancer.gov/projects?filters=%7B%22op%22%3A%22and%22%2C%22content%22%3A%5B%7B%22op%22%3A%22in%22%2C%22content%22%3A%7B%22field%22%3A%22projects.program.name%22%2C%22value%22%3A%5B%22TCGA%22%5D%7D%7D%5D%7D)
- Install download software: [GDC Data Transfer Tool](https://gdc.cancer.gov/access-data/gdc-data-transfer-tool)
- Downloading slides using the manifest File of each disease type:
  ```python
  gdc-client download -m gdc_manifest.<cancer-type>.txt
  ```


## Download slides from CPTAC

### Downloading TCIA Faspex Packages Using Aspera CLI

The [Cancer Imaging Archive (TCIA)](https://wiki.cancerimagingarchive.net/) contains various datasets, including the Clinical Proteomic Tumor Analysis Consortium (CPTAC) pathology slide collections. 
These collections offer bulk downloads of pathology slides for specific cancer types through Aspera download packages.

### Prerequisites

Ensure you have Ruby and the Aspera CLI installed on your system:

```sh
# Install Ruby and its development environment
sudo apt-get install ruby ruby-dev rubygems

# Install the aspera-cli gem
gem install aspera-cli

# Install the Aspera Connect client via ascli
ascli conf ascp install
```

### Downloading Aspera Packages

To download a package from TCIA using ascli, perform the following steps:

- Visit the [CPTAC Pathology Slide Downloads page](https://wiki.cancerimagingarchive.net/display/Public/CPTAC+Pathology+Slide+Downloads)
- Choose the specific cancer type collection you wish to download (e.g., CPTAC-AML, CPTAC-CCRCC).
- Use the copied URI in the following ascli command to initiate the download:
  ```sh
  # Replace <PACKAGE_URI> with the actual link and <TARGET_DIRECTORY> with your desired download location.
  ascli faspex package recv --url=<PACKAGE_URI> --to-folder=<TARGET_DIRECTORY>
  ```
  For example, to download the CPTAC-AML package:
  ```sh
  ascli faspex package recv --url=https://faspex.cancerimagingarchive.net/aspera/faspex/... --to-folder=/path/to/download/directory
  ```
  




## Tile slides
```python
python wsi_tiling.py  \
  --dataset <path of dataset containing svs>  \
  --output <path to store tiled patches>  \
  --scale 20 --patch_size 1024 --num_threads 16
```

