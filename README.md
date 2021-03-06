### SynergyY

SynergyY is an all-in-one package for drug synergy prediction. This package allows the user to conduct **standardized** experiments to compare the prediction performance between reviewed methods.  

The user can freely include new datasets, and select preferential cell/drug features to train the deep learning model.
****

#### Installation

```bash
# Unzip 
unzip synergyy.zip
cd synergyy/

#create conda environment
conda env create --name synergyy --file=environment.yml
conda activate synergyy
#To install for PyTorch 1.10.0, simply run on your mac
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.10.0+${cpu}.html
pip install torch-geometric -f https://pytorch-geometric.com/whl/torch-1.10.0+${cpu}.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.10.0+${cpu}.html 
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.10.0+${cpu}.html 
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.10.0+${cpu}.html

#install synergyy
pip install -e .

#Please download data/ and put it in the same path as setup.py
https://drive.google.com/drive/folders/1Uu0YZSxX8GQtV_4ZJmsMrbanmse-Dq6n?usp=sharing

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
| ML approaches: LR,RF,XGBoost,ERT  | exp or cnv or mut  |  Drug-target interaction    |       |      |         |       |  True|        
|  DeepSynergy   |    exp  |  Drug chemical descriptor or fingerprints    |    DNN             |   DNN    |      |      |   True   |
|  MatchMaker    |    exp  |  Drug chemical descriptor or fingerprints   |    DNN              |  DNN    |      |      |   False    |
|  Multitask_DNN |    exp  |  Morgan or MACCS fingerprints, Drug-target interaction    |     DNN   | DNN     |  | False| False |
| Graph based: DeepDDS|    exp |  SMILES2Graph   |     MLP   |   GCN   | | | False |
| Graph based: TGSynergy (modified from TGSA)|    exp |  SMILES2Graph   |     GCN   |   GCN   | | | False |
| Graph based: TranSynergy|    exp |  Network propagated Drug-target interaction   |     Transformer   |   GCN(RWR)+Transformer   | | | False |
****
#### Data downloaded
SynergyY used multi-omics datasets. 
1. We have provided a cleaned benchmark DrugComb v1.5 synergy truset. For details of reporducing, please go to trueset_generation/ to follow the instructions.
2. CCLE dataset including exp, cnv, mut
3. Drug-target interaction dataset from DrugComb, and structures.sdf which  enables fingerprints calculation or smiles2graph
****

#### Models included (on-the-fly)
In detail, the following drug synergy prediction models were implemented.
- Baseline machine Learning models (random forest, extreme gradient boosting, extremely randomized tree, logistic regression)

- End-to-end deep learning models
    - [1] [Kristina Preuer, Richard PI Lewis, Sepp Hochre-iter, Andreas Bender, Krishna C Bulusu, and G ??unter Klambauer.DeepSynergy: Predicting Anti-Cancer Drug Synergy with DeepLearning.Bioinformatics, 34(9):1538???1546, 2018.](https://academic.oup.com/bioinformatics/article/34/9/1538/4747884?login=false)
    - [2] [Kuru Halil Brahim, Oznur Tastan, and Ercument Cicek. MatchMaker: A Deep Learning Framework for Drug Synergy Prediction. IEEE/ACM Transactions on Computational Biology and Bioinformatics, 2021.](https://ieeexplore-ieee-org.proxy.library.cornell.edu/document/9447196/)
    - [3] [Yejin Kim, Shuyu Zheng, Jing Tang, Wenjin Jim Zheng, Zhao Li, and Xiaoqian Jiang. Anticancer Drug Synergy Prediction in Understudied Tissues Using Transfer Learning. Journal of the American Medical Informatics Association, 28(1):42???51, 2021.](https://academic.oup.com/jamia/article/28/1/42/5920819?login=true)
    - [4] [Jinxian Wang, Xuejun Liu, Siyuan Shen, Lei Deng, and Hui Liu. DeepDDS: Deep Graph Neural Network with Attention Mechanism to Predict Synergistic Drug Combinations. Briefings in Bioinformatics, 09 2021](https://academic.oup.com/bib/article/23/1/bbab390/6375262)
    - [5] [Yiheng Zhu, Zhenqiu Ouyang, Wenbo Chen, Ruiwei Feng, Danny Z Chen, Ji Cao, Jian Wu, TGSA: protein???protein association-based twin graph neural networks for drug response prediction with similarity augmentation, Bioinformatics, Volume 38, Issue 2, 15 January 2022, Pages 461???468](https://academic.oup.com/bioinformatics/article-abstract/38/2/461/6374919?redirectedFrom=fulltext)
    - [6] [Liu, Qiao, and Lei Xie. "TranSynergy: Mechanism-driven interpretable deep neural network for the synergistic prediction and pathway deconvolution of drug combinations." PLoS computational biology 17.2 (2021): e1008653.](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008653&ref=https://githubhelp.com)
****

#### Constructing ...
We'll include more deep learning models on-the-fly.  
We'll implement algorithms to estimate gene importances from the models.
****
#### License
[MIT](https://choosealicense.com/licenses/mit/)
