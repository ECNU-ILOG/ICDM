# :sparkles: :sparkles: <u>ICDM</u>-WWW 2024 :sparkles::sparkles:

:smile_cat: Welcome to ICDM, this is a comprehensive repository specializing in ***Inductive Cognitive Diagnosis for Fast Student Learning in Web-Based Online Intelligent Education Systems*** published in WWW 2024.

We provide comprehensive instructions on how to run ICDM in the ***<u>"exps"</u>*** directory. If you're interested, please navigate to the exps directory for more information.

# Requirements	

```python
dgl==1.1.2+cu116
EduCDM==0.0.13
joblib==1.2.0
networkx==2.6.3
numpy==1.23.5
pandas==1.5.2
scikit_learn==1.2.3
scipy==1.12.0
torch==1.13.1+cu117
tqdm==4.65.0
wandb==0.15.2
```

> cd exps

## Transductive Scenario

```shell
python exp.py --exp_type=cdm --method=icdm --datatype=EdNet-1 --test_size=0.2 --seed=0 --dim=64 --epoch=8 --device=cuda:0 --gcnlayers=3 --lr=1e-3 --agg_type=mean --cdm_type=glif --khop=1
```



## Inductive Scenario 

```shell
python exp_ind.py  --exp_type=ind --method=icdm --datatype=EdNet-1 --test_size=0.2 --seed=0 --dim=64 --epoch=5 --device=cuda:0 --gcnlayers=3 --lr=2e-3 --agg_type=mean --mode=train --cdm_type=glif --new_ratio=0.2 --khop=3
```



# Experiment :clap:

<u>We utilize **wandb**, a practical and effective package for visualizing our results. However, if you prefer not to use it, it can be easily disabled.</u> https://wandb.ai/ :scroll:

# Reference :thought_balloon:

Shuo Liu, Junhao Shen, Hong Qian, Aimin Zhou "Inductive Cognitive Diagnosis for Fast Student Learning in Web-Based Online Intelligent Education Systems." In Proceedings of the ACM Web Conference, 2024.

## Bibtex

```
@inproceedings{liu2024www,
author = {Shuo Liu, Junhao Shen, Hong Qian, Aimin Zhou},
booktitle = {Proceedings of the ACM Web Conference 2024},
title = {Inductive Cognitive Diagnosis for Fast Student Learning in Web-Based Online Intelligent Education Systems},
year = {2024},
address={Singapore}
}
```
