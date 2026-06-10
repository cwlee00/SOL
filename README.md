# [ICML 2026] SOL

Chaewon Lee,
Seon-Ho Lee,
and Chang-Su Kim

Official code for **"Stochastic Order Learning: An Approach to Rank Estimation Using Noisy Data"**

### Requirements
- PyTorch 2.3.0
- torchvision 0.18.0
- CUDA 11.8
- cuDNN 8.7
- python 3.9
  
### Installation
Create conda environment:
```bash
    $ conda env create -f environment.yaml -n SOL
    $ conda activate SOL
```
Download repository:
```bash
    $ git clone https://github.com/cwlee00/SOL.git
```
Download weights:

SOL model [Google Drive](https://drive.google.com/drive/folders/1d1YdM6WcX3b0yIFAPFNP_SJP3CF1XllU)

### Evaluation
For evaluation, please download the datasets and models, and then configure the path in [config.yml](https://github.com/cwlee00/SOL/tree/main/config)

```
python test.py \
--checkpoint=./weights/SOL_models/SOL(CLAP_Gaussian20).pth \
--dataset=clap
```
### Train
For training, please download the datasets, and then configure the path in [config.yml](https://github.com/cwlee00/SOL/tree/main/config)
```
python train.py \
--dataset=clap \
--noise_type=Gaussian \
--noise_percentage=20 \
```

### Citation
Please cite the following paper if you feel this repository useful.
```bibtex
    @InProceedings{Lee_2026_SOL_ICML,
    author    = {Lee, Chaewon and Lee, Seon-Ho and Kim, Chang-Su},
    title     = {Stochastic Order Learning: An Approach to Rank Estimation Using Noisy Data},
    booktitle = {Forty-third International Conference on Machine Learning},
    year      = {2026}
    }
```

