# Wavoice
Pytorch implementation of our proposed system for noise-resistant multi-modal speech recognition system via fusing mmWave and audio signals.

Wavoice: A Noise-resistant Multi-modal Speech Recognition System Fusing mmWave and Audio Signals.

Tiantian Liu,Gao Ming,Chao Wang,Feng Lin,Zhongjie Ba,Jinsong Han,Wenyao Xu,Kui Ren in SenSys2021

## Prerequisites
- Linux
- Python 3.8
- NVIDIA GPU

## Getting Started
### Installation
- Install PyTorch 1.8.0 and dependencies from http://pytorch.org
- Install some python libraries
```bash
git clone https://github.com/TitaniumLiu/Wavoice.git
pip install -r requirements.txt
```
### Preprocessing
Before training the model, please preprocess the mmWave and speech signal.The dataset comprises: train_mmwave,train_voice,test_mmwave,and test_voice. Please run the stript `util/prepare_Wavoice.py` after seting the right path to the dataset in the stript. This will create a `processed/` folder containing three csvs files that save training paths, testing paths,and character label files  

### Training
- Modify parameters in `config/Wavoice-config.yaml` to train and test the model. Make sure the path to the `processed/` right in the `-config.yaml`.
- Tain a model
```bash
python train.py --config_path config/Wavoice-config.yaml
```
### Training with your own dataset
Please prepare the mmWave/audio dataset and the corresponding groundtruth according to [LibriSpeech](https://www.openslr.org/12)

## Citation
If you find this useful for your research, please use the following.
```
@inproceedings{liu2021Wavoice,
  title={A Noise-resistant Multi-modal Speech Recognition System Fusing mmWave and Audio Signals},
  author={Tiantian Liu,Gao Ming,Chao Wang,Feng Lin,Zhongjie Ba,Jinsong Han,Wenyao Xu,Kui Ren}, 
  booktitle={Proceedings of the 19th Conference on Embedded Networked Sensor Systems (SenSys)},
  year={2021}
}
```
## Reference 
The codes for Lisen, Attend, and Speller(LAS) in the system is borrowed from https://github.com/jiwidi/las-pytorch

