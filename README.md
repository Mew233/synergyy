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

conda activate synergyy

#install synergyy
pip install -e .

#Please download data/ and put it in the same path as setup.py
#https://drive.google.com/drive/folders/1Uu0YZSxX8GQtV_4ZJmsMrbanmse-Dq6n?usp=sharing

```
****

#### Getting strarted

```bash
 python synergyy/main.py --model 'deepsynergy_preuer' --synergy_df 'DrugComb' --train_test_mode train
```

#### Features explained
| Model |  Input feature format      || Feature encoders       || Features concatenated   ||Drug1 and drug2 summed  |
| ------|:--------------------:|:----:|:----------------:|:----:|:----------------:|:----:|:----------------:|
|       | *Cell line*            | *Drug* | *Cell line*        | *Drug* |     *Cell line*   |       *Drug*               |      
| ML approaches: RF,XGBoost,ERT  | exp/cnv/mut  |  Drug-target interaction    |       |      |         |       |  True|        
|  DeepSynergy   |    exp  |  Drug chemical descriptor    |    DNN             |   DNN    |      |      |   True   |
|  MatchMaker    |    exp  |  Drug chemical descriptor    |    DNN              |  DNN    |      |      |   False    |
|  Multitask_DNN |    exp, Cancer/Tissue type|  MACCS fingerprints, SMILES, Drug-target interaction    |     DNN   | DNN     | False | False| False |
|  DeepDDS|    exp |  SMILES   |     GCN   | MLP     | | | False |

****
#### Data downloaded
SynergyY used multi-omics datasets. 
1. DrugComb v1.5 synergy dataset 
2. CCLE dataset including exp, cnv, mut
3. Drug-target interaction dataset from DrugComb,  and structures.sdf which  enables fingerprints calculation
****

#### Models included (on-the-fly)
In detail, the following drug synergy prediction models were implemented.
- Baseline machine Learning models (random forest, extreme gradient boosting, extremely randomized tree, logistic regression)

- End-to-end deep learning models
    - [1] [Kristina Preuer, Richard PI Lewis, Sepp Hochre-iter, Andreas Bender, Krishna C Bulusu, and G ̈unter Klambauer.DeepSynergy: Predicting Anti-Cancer Drug Synergy with DeepLearning.Bioinformatics, 34(9):1538–1546, 2018.](https://academic.oup.com/bioinformatics/article/34/9/1538/4747884?login=false)
    - [2] [Kuru Halil Brahim, Oznur Tastan, and Ercument Cicek. MatchMaker: A Deep Learning Framework 
    for Drug Synergy Prediction. IEEE/ACM Transactions on Computational Biology and Bioinformatics, 2021.](https://ieeexplore-ieee-org.proxy.library.cornell.edu/document/9447196/)
    - [3] [Yejin Kim, Shuyu Zheng, Jing Tang, Wenjin Jim Zheng, Zhao Li, and Xiaoqian Jiang. Anticancer Drug Synergy
    Prediction in Understudied Tissues Using Transfer Learning. Journal of the American Medical Informatics Association, 28(1):42–51, 2021.](https://academic.oup.com/jamia/article/28/1/42/5920819?login=true)
    - [4] 
****

#### Constructing ...
We'll include more models.  
We'll include Grad-Cam to identify the important genes from the last CNN layer.
****
#### License
[MIT](https://choosealicense.com/licenses/mit/)
