# GeoStyler
(2026 TMM) Optimal Geometric Style Distribution with Adaptive Inference for Zero-Shot Domain Generalization

## How to run

### Installation

The following setup uses Python 3.8, PyTorch 1.13.1, and CUDA 11.7 on Linux with an NVIDIA GPU.

```bash
git clone https://github.com/2hlovely/GeoStyler.git
cd GeoStyler
conda create -n geostyler python=3.8 -y
conda activate geostyler
python -m pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
python -m pip install -r requirements.txt
```

### Dataset preparation

Place the datasets under a common root directory and use the layouts expected by the dataset readers:

| Dataset | Directory layout under `DATASET_ROOT` |
| --- | --- |
| PACS | `pacs/images/<domain>/<class>/` and `pacs/splits/<domain>_{train,crossval}_kfold.txt` |
| VLCS | `VLCS/<domain>/{train,crossval,test}/<class>/` |
| OfficeHome | `office_home_dg/<domain>/{train,val}/<class>/` |
| DomainNet | `domainnet/images/<domain>/<class>/` and `domainnet/splits/<domain>_{train,test}.txt` |

Set `DATASET_ROOT` to this root directory and `DEVICE` to the GPU index in both the training and test scripts.

The scripts use ResNet-50 by default. Set `NET=vitb16_clip` in both scripts to run the ViT-B/16 experiments.

### Training and evaluation

Run the training script first, followed by the corresponding test script:

```bash
bash scripts/dataset_train.sh
bash scripts/dataset_test.sh
```

Logs and checkpoints are saved under `output/`.

# Acknowledgement
This project is built on the [Dassl](https://github.com/KaiyangZhou/Dassl.pytorch) framework and references the open-source code of [DPStyler](https://github.com/TYLfromSEU/DPStyler). Special thanks are extended to the authors of both projects.

## Citation

If you find our work helpful, please consider citing our paper:

```bibtex
@ARTICLE{ma26geostyler,
  author={Ma, Rui and Zhao, Boyu and Wang, Yisong and Jin, Yutang and Duan, Bowen and Xie, Guo-Sen and Chen, Shiming},
  journal={IEEE Transactions on Multimedia},
  title={Optimal Geometric Style Distribution With Adaptive Inference for Zero-Shot Domain Generalization},
  year={2026},
  volume={28},
  pages={18509-18521}
}
```
