# Parallel Optimal Deep Echo State Network
## Introduction
This repository is a generic implementation of "Parallel Optimal Deep Echo State Network", part of the implementation of "LEOPARD: Parallel Optimal Deep Echo State Network Prediction Improves Service Coverage for UAV-Assisted Outdoor Hotspots" in IEEE Transactions on Cognitive Communication and Network. [[Paper]](https://ieeexplore.ieee.org/document/9548955) [[Slides]](https://haoran-peng.github.io/Slides/LEOPARD_TCCN.pdf) 
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
