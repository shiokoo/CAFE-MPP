# CAFE-MPP
This is the official implmentation of CAFE-MPP.  
## Setup
### Installation
    # conda environment
    conda create --name CAFE-MPP python=3.8
    conda activate CAFE-MPP

    # install requirements
    conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
    conda install pyg -c pyg
    conda install -c conda-forge rdkit
    conda install -c anaconda cython

    # clone the source code
    git clone https://github.com/shiokoo/CAFE-MPP.git
    cd CAFE-MPP

### Dataset
We provide the preprocessed pre-training fragment dataset used in this project. Besides, You can download the benchmarks ***MoleculeNet*** used in this project by running the following command:

    cd ./Data
    bash download_data.sh

### Pre-training
To pre-train the CAFE-MPP, where the configurations and hyperparameters are defined in `./Config/config_pretrain.yaml`.

    cd ./Pretrain
    python trainer.py

### Prediction
To fine-tune the CAFE-MPP, where the configurations and details are can be found in `./Config/config_prediction.yaml`.

    cd ./Prediction
    python trainer.py



