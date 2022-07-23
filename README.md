# LEOPARD: Parallel Optimal Deep Echo State Network Prediction Improves Service Coverage for UAV-Assisted Outdoor Hotspots
## Introduction
This repository is a generic implementation and example of "Parallel Optimal Deep Echo State Network", the system architecture was presented in "LEOPARD: Parallel Optimal Deep Echo State Network Prediction Improves Service Coverage for UAV-Assisted Outdoor Hotspots" in IEEE Transactions on Cognitive Communication and Network. [[Paper]](https://ieeexplore.ieee.org/document/9548955) [[Slides]](https://haoran-peng.github.io/Slides/LEOPARD_TCCN.pdf) 
You can train your own model using this repository.

The implementation of DeepESN is based on [lucapedrelli-DeepESN](https://github.com/lucapedrelli/DeepESN) and [Xeeshanmalik-deep_ml_esn](https://github.com/Xeeshanmalik/deep_ml_esn).

The architecture of parallel computing is based on [zblanks-parallel_esn](https://github.com/zblanks/parallel_esn).

> There are some limitations to this work. If you have any questions or suggestions, please feel free to contact me. Your suggestions are greatly appreciated.

## Citing
Please consider **citing** our paper if this repository is helpful to you.
**Bibtex:**
```
@INPROCEEDINGS{9548955,
  author={Peng, Haoran and Tsai, Ang-Hsun and Wang, Li-Chun and Han, Zhu},
  booktitle={IEEE Trans. Cogn. Commun. Netw.}, 
  title={{LEOPARD}: Parallel Optimal Deep Echo State Network Prediction Improves Service Coverage for {UAV}-Assisted Outdoor Hotspots},
  volume={8},
  number={1},
  pages={282--295},
  year={2022},
  month = {Mar.}
}
```
## Requirements
- Python: 3.6.13
- xlwt
- bayesian-optimization
- numpy: <= 1.17
- matplotlib
- pandas

## Usage
#### Descriptions of the data files
- The files "lat" and "lon" are the training data sampled from the [GeoLife GPS Trajectory Dataset](https://www.microsoft.com/en-us/download/details.aspx?id=52367&from=https%3A%2F%2Fresearch.microsoft.com%2Fen-us%2Fdownloads%2Fb16d359d-d164-469e-9fd4-daa38f2b2e13%2F). The last 100 data samples in each file are used for testing and prediction.
- The files "lat_result.xls" and "lon_result.xls" are the predicted results.
- The files "params_lat" and "params_lon" are the optimized paramaters find by Bayesian Optimization.
- The files "deepESN_loss_lat" and "deepESN_loss_lon" are the recorded loss during the training process.

#### Descriptions of python files
The deep echo state network is impletemented in 'DeepESN.py'.
The parallel computing and Bayesian Optimization are implemented in 'train.py'.
The file 'predict.py' is for testing.

#### Training phase
1. For training model by the data lat.txt
```
python train.py lat
```
2. For training model by the data lon.txt
```
python train.py lon
```
#### Testing phase
1. For testing model using the data lat.txt
```
python predict.py lat
```
2. For testing model using the data lon.txt
```
python predict.py lon
```
