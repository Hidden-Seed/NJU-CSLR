# Chinese Sign Language Recognition(NJU-CSLR)

## Project Introduction

+ This project belongs to Chinese Sign Language Recognition Research Group, School of Electronic Science and Engineering, **Nanjing University** and is intended for **non-commercial purposes only**, such as teaching and demonstration.
+ The members of this project:
    - **Project Leader**: Shijie Sun
    - **Team Members**: Chao Xu, Yuankai Lai
    - **Supervisors**: Hongchen Zhan, Yi Chen, Yina Zhou
+ The goal of this project: **Chinese Isolated Sign Language Word Recognition based on PyTorch**;
+ The basic network structure: **Bi-LSTM** (Bidirectional Long Short-Term Memory Neural Network);
+ The dataset: **500-CSL** provided by the University of Science and Technology of China ([CSL & SLR500 Dataset | Visual Sign Language Research Group](https://ustc-slr.github.io/datasets/2015_csl/)).
+ This project references the following project: [harveyfly/SignLanguageRecognition: 中文孤立手语词识别；Bi-LSTM；SLR；500-CSL](https://github.com/harveyfly/SignLanguageRecognition)

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
├── config                 # config files (model, dataset and etc.)
├── dataset
│   └── dictionary.txt     # labels of dataset 
├── nnet                   # neural networks
│   ├── __init__.py
│   ├── blstm.py           # ai model based on blstm
│   ├── lstm.py            # ai model based on lstm     
│   └── model.py           # other models
├── utils                  # necessary modules for data processing, training and etc.
├── mp_hlt.py              # color_video to txt_file (mediapipe holistic)
├── txt2mat.py             # txt_file to mat_data
├── data_preprocess.py     # preprocess mat_data
├── data_split_shuffle.py  # split and shuffle
├── skeleton_viz.py        # visualize processed data
├── train.py               # training for blstm/lstm
├── train_model.py         # training for other models
├── train.sh               # start training
├── prediction             # several demos
├── prediction.py          # test model with demos in prediction_dir
├── run_demo.sh            # start prediction
├── README.MD
└── requirements.txt 
```

## Workflow

### Modify the configuration files

```bash
config/
├── dataset.cfg  # Dataset configuration
├── model.cfg    # Model configuration
├── mp.cfg       # Mediapipe holistic configuration
└── server.cfg   # Server configuration (demo application)
```

### Download the dataset and process it

**Download**

+ Apply to the USTC [CSL & SLR500 Dataset | Visual Sign Language Research Group](https://ustc-slr.github.io/datasets/2015_csl/)

**Process**

```bash
$ # Modify the data path before running the it
$ python mp_hlt.py
$ python txt2mat.py
```

### Data process, train and test

```bash
$ ./train.sh  # include preprocess & split_shuffle
```

Training and testing results are recorded in the log file (`log/`), and will also be displayed on the terminal. The model files are saved in the `model/` folder.

### Run the demo application

```bash
$ ./run_demo.sh
```

## Copyright

Copyright © 2025 Chinese Sign Language Recognition Research Group, School of Electronic Science and Engineering, Nanjing University. All rights reserved.
This project is intended for research and educational purposes only. Unauthorized use, reproduction, or distribution of this work is prohibited.
