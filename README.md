# HMGNN
This is code for our ECMLPKDD2022 paper《Heterogeneous Multigraph Neural Network with Supervised Contrastive Learning for Friend Recommendation in Location-based Social Networks》


Before to execute *HMGNN*, it is necessary to install the following packages:
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

### Data Set

Due to the size limitation of Supplementary Material, we cannot add the processed data here.

At the same time, due to the limitations of the Double-blind Reviewing Process, current links such as Google's drive often carry personal information, so we feel that it is unsuitable to pass data through these drive links.

Therefore, we have decided to only open the code at this stage. If the paper be accepted, we will publish all the data and other information on Github.  You can be sure that we'll try our best!

You can download whole [raw Foursquare Dataset](https://sites.google.com/site/yangdingqi/home/foursquare-dataset) here.

### Basic Usage
 
- --run  main.py to train the HMGNN. and it probably need at least 11G GPU memory 
- --run  test.py to estimate the performance of HMGNN based on the user representations that we learned during our experiments. You can also use this code to individually test the effects of your own learned representation.

### Miscellaneous

*Note:* This is only a reference implementation of *HMGNN*. Our code implementation is partially based on the DGL library, for which we are grateful.
