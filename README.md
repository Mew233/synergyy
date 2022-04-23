### SynergyY

SynergyY is an all-in-one package for drug synergy prediction. This package allows the user to conduct **unbiased** experiments to compare the prediction performance between reviewed methods.  

The user can also freely to include new dataset , and select preferential cell/drug features to train the deep learning model.
****

#### Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install synergyy.

```bash
# Unzip 
unzip synergyy.zip

cd synergyy/

#create conda environment
conda env create --name synergyy --file=environments.yml

#install synergyy
pip install -e .

```
****

#### Getting strarted

```bash
 python synergyy/main.py --model 'deepsynergy_preuer' --synergy_df 'DrugComb' --train_test_mode train
```

#### Features explained
| Model |  Input feature format      || Feature encoders       || Dataset | Performance measures | Ref. |
| ------|:--------------------:|:----:|:----------------:|:----:|:-------:|:--------------------:|-----:|
|       | Cell line            | Drug | Cell line        | Drug |         |                      |      |
|       |                      |      |                  |      |         |                      |      |
|       |                      |      |                  |      |         |                      |      |
|       |                      |      |                  |      |         |                      |      |

****
#### Data downloaded
SynergyY used multi-omics datasets. Please download:  
1. DrugComb v1.5 synergy dataset 
2. CCLE dataset including exp, cnv, mut
3. Drug-target interaction dataset from DrugComb,  and structures.sdf which  enables fingerprints calculation
****

#### Models included (on-the-fly)
In detail, the following drug synergy prediction models were implemented.
- Baseline machine Learning models (random forest, extreme gradient boosting, extremely randomized tree, logistic regression)

- End-to-end deep learning models
    - [1] Kristina Preuer, Richard PI Lewis, Sepp Hochre-iter, Andreas Bender, Krishna C Bulusu, and G ̈unter Klambauer.DeepSynergy: Predicting Anti-Cancer Drug Synergy with DeepLearning.Bioinformatics, 34(9):1538–1546, 2018.
    - [2] Kuru Halil Brahim, Oznur Tastan, and Ercument Cicek. MatchMaker: A Deep Learning Framework 
    for Drug Synergy Prediction. IEEE/ACM Transactions on Computational Biology and Bioinformatics, 2021.
    - [3] Yejin Kim, Shuyu Zheng, Jing Tang, Wenjin Jim Zheng, Zhao Li, and Xiaoqian Jiang. Anticancer Drug Synergy
    Prediction in Understudied Tissues Using Transfer Learning. Journal of the American Medical Informatics Association, 28(1):42–51, 2021.
****

#### Constructing ...
We'll include more models.  
We'll include Grad-Cam to identify the important genes from the last CNN layer.
****
#### License
[MIT](https://choosealicense.com/licenses/mit/)
