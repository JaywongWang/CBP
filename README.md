# CBP
Official Tensorflow Implementation of the AAAI-2020 paper [Temporally Grounding Language Queries in Videos by Contextual Boundary-aware Prediction](https://arxiv.org/abs/1909.05010) by Jingwen Wang *et al.*

![alt text](method.png)

### Requirements
python 2.7
``` bash
pip install -r requirements.txt
```

### Data Preparation
1. Download Glove word embedding data.
``` shell
cd download/
sh download_glove.sh
```

2. Download dataset features.

[TACoS](https://drive.google.com/file/d/13JLnFhSzi8MPRzOG2Ao_q-J5-T5tewcg/view?usp=sharing)

[Charades-STA](https://pan.baidu.com/s/1ODW4JIXfCCIbozPcaD_-UA)

[ActivityNet-Captions](https://pan.baidu.com/s/1W9S7_nHf3nzDm1TDjm0YBA)

Put the feature hdf5 file in the corresponding directory `./datasets/{DATASET}/features/`

We decode TACoS/Charades videos using `fps=16` and extract C3D (fc6) features for each non-overlap 16-frame snippet. Therefore, each feature corresponds to 1-second snippet. For ActivityNet, each feature corresponds to 2-second snippet.

3. Download trained models.

Download and put the checkpoints in corresponding `./checkpoints/{DATASET}/` .

[TACoS](https://drive.google.com/file/d/1cyja-U3weuo7CDYhLMMr511Yn1SiXRnc/view?usp=sharing)

[Charades-STA](https://drive.google.com/file/d/1eKupvkgD2s9ViFltXF6KPVAZr0Nu5XGu/view?usp=sharing)

[ActivityNet-Captions](https://drive.google.com/file/d/11FEUaH4Vd9TGcFaowOp4PD9Kn-GVWHEP/view?usp=sharing)


4. Data Preprocessing (Optional)
``` shell
cd datasets/tacos/
sh prepare_data.sh
```
Then copy the generated data in `./data/save/` .

Use correspondig scripts for preparing data for other datasets.

You may skip this procedure as the prepared data is already saved in `./datasets/{DATASET}/data/save/` .

### Testing and Evaluation

``` shell
sh scripts/test_tacos.sh
sh scripts/eval_tacos.sh
```
Use corresponding scripts for testing or evaluating for other datasets.

The predicted results are also provided in `./results/{DATASET}/` .

### Training

``` shell
sh scripts/train_tacos.sh
```
Use corresponding scripts for training for other datasets.