# Chinese Sign Language Recognition(CSLR)

## Project Introduction

+ The members of this project:
    - **Project Leader**: Shijie Sun
    - **Team Members**: Chao Xu, Yuankai Lai
    - **Supervisor**: Hongchen Zhan, Yi Chen, Yina Zhou
+ The goal of this project: **Chinese Isolated Sign Language Word Recognition based on PyTorch**;
+ The basic network structure: **Bi-LSTM** (Bidirectional Long Short-Term Memory Neural Network);
+ The dataset: **500-CSL** provided by the University of Science and Technology of China ([500-class Chinese Sign Language Word Dataset](http://home.ustc.edu.cn/~pjh/openresources/cslr-dataset-2015/index.html)).

## Environment configuration

+ Ubuntn  24.04
+ python  3.10.16
+ PyTorch 2.5.1+cu121

**Install packages**
```bash
$ pip install -r requirements.txt
```

## Project file tree

```bash
CSLR/
├── config
│   ├── dataset.cfg
│   ├── __init__.py
│   ├── model.cfg
│   └── server.cfg
├── CSLR_client.py
├── CSLR_flask_server.py
├── data_preprocess.py
├── dataset
│   └── dictionary.txt
├── data_split_shuffle.py
├── nnet
│   ├── AutoEncoder.py
│   ├── blstm.py
│   ├── __init__.py
│   └── lstm.py
├── README.MD
├── requirements.txt
├── train.py
├── train.sh
├── txt2mat.m
└── utils
    ├── data_config.py
    ├── __init__.py
    ├── keyframes.py
    ├── logger.py
    ├── plot_data.py
    ├── server_config.py
    ├── train_config.py
    └── utils.py
```

## Workflow

### Download the dataset and convert it

**Download**

+ Use [harveyfly's dataset backup](https://pan.baidu.com/s/1lZasotYbZpsoCrilvT71VQ) (which has been converted to MAT files), extraction code: vswo.
+ Apply to the USTC [500-class Chinese Sign Language Word Dataset](http://home.ustc.edu.cn/~pjh/openresources/cslr-dataset-2015/index.html).

**Conversion**

```bash
$ # Modify the data path before running the it
$ matlab -batch txt2mat.m
```

### Modify the configuration files

```bash
./config/
├── dataset.cfg  # Dataset configuration
├── model.cfg    # Model configuration
└── server.cfg   # Server configuration (demo application)
```
### Data process, train and test

```bash
$ ./train.sh
```

Training and testing results are recorded in the log file (`log/`), and will also be displayed on the terminal. The model files are saved in the `model/` folder.

### Run the demo application

```bash
$ python CSLR_flask_server.py
$ python CSLR_client.py
```

Copyright © 2025 Chinese Sign Language Recognition Research Group, School of Electronic Science and Engineering, Nanjing University. All rights reserved.
This project is intended for research and educational purposes only. Unauthorized use, reproduction, or distribution of this work is prohibited.
