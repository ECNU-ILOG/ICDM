# :sparkles: :sparkles: <u>ICDM</u> :sparkles::sparkles:

:smile_cat: Welcome to **ICDM, this is a comprehensive repository specializing in ***Inductive Cognitive Diagnosis for Fast Student Learning\\ in Web-Based Online Intelligent Education Systems***.

------

## ICDM-*WWW2024*

We provide `ICDM-WWW2024.pdf`  in "*<u>**papers**</u>*" directory.

We provide comprehensive instructions on how to run ICDM in the ***<u>"exps/ICDM"</u>*** directory. If you're interested, please navigate to the exps/ICDM directory for more information.

> Implementation of Inductive Cognitive Diagnosis for Fast Student Learning in Web-Based Online Intelligent Education Systems

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

Shuo Liu, Junhao Shen, Hong Qian, Aimin Zhou "Inductive Cognitive Diagnosis for Fast Student Learning in Web-Based Online Intelligent Education Systems." In Proceedings of the The Web Conference, 2024.

## Bibtex

```
@inproceedings{liu2024www,
author = {Shuo Liu, Junhao Shen, Hong Qian, Aimin Zhou},
booktitle = {Proceedings of the The Web Conference 2024},
title = {Inductive Cognitive Diagnosis for Fast Student Learning in Web-Based Online Intelligent Education Systems},
year = {2024},
address={Singapore}
}
```
