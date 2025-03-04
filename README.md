# HMGCL
This is code for our WWWJ paper《HMGCL: Heterogeneous Multigraph Contrastive Learning for LBSN Friend Recommendation》


Before to run *HMGCL*, it is necessary to install the following packages:
<br/>
``pip install dgl``
<br/>
``pip install torch``
<br/>
``pip install scikit-learn``

## Requirements

- numpy ==1.13.1
- torch ==1.7.1
- scikit-learn==1.0.2
- dgl ==0.7.2

【September 11, 2024】The modified code now supports the latest versions of torch and DGL.

### Data Set


You can download whole [raw Foursquare Dataset](https://sites.google.com/site/yangdingqi/home/foursquare-dataset) here.

Our data can be found at [here](https://drive.google.com/file/d/1i6W2oz0PEidhG2md6pn1u-HRT3wmWyh7/view?usp=sharing).

【March 3, 2025】 Please refer to the code implementation in [H3GNN](https://github.com/liyongkang123/H3GNN) to learn more about processing heterogeneous multigraph data. 
We provide an updated version of the code that can complete the data preprocessing for a single city within 10 minutes.

### Basic Usage
 
- --run  main.py to train the HMGCL. and it probably need at least 11G GPU memory 
- --run  test.py to estimate the performance of HMGCL based on the user representations that we learned during our experiments. You can also use this code to individually test the effects of your own learned representation.

### Miscellaneous

*Note:* This is only a reference implementation of *HMGCL*. Our code implementation is partially based on the DGL library, for which we are grateful.

# Citation
If you find this work helpful, please consider citing our paper:
```bibtex
@article{li2023hmgcl,
  title={HMGCL: Heterogeneous multigraph contrastive learning for LBSN friend recommendation},
  author={Li, Yongkang and Fan, Zipei and Yin, Du and Jiang, Renhe and Deng, Jinliang and Song, Xuan},
  journal={World Wide Web},
  volume={26},
  number={4},
  pages={1625--1648},
  year={2023},
  publisher={Springer},
  url = {https://doi.org/10.1007/s11280-022-01092-5},
  doi = {10.1007/s11280-022-01092-5},
}
```