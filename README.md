# :sparkles: :sparkles: <u>ICDM</u> :sparkles::sparkles:

:smile_cat: Welcome to **ICDM, this is a comprehensive repository specializing in ***Inductive Cognitive Diagnosis for Fast Student Learning\\ in Web-Based Online Intelligent Education Systems***.

------

## ICDM-*WWW2024*

We provide `ICDM-WWW2024.pdf`  in "*<u>**method/ICDM/papers**</u>*" directory.

We provide comprehensive instructions on how to run ICDM in the ***<u>"exps/ICDM"</u>*** directory. If you're interested, please navigate to the exps/ICDM directory for more information.

> Implementation of Inductive Cognitive Diagnosis for Fast Student Learning\\ in Web-Based Online Intelligent Education Systems

> cd exps

## Transductive Scenario

```shell
python exp.py --exp_type=cdm --method=icdm --datatype=EdNet-1 --test_size=0.2 --seed=0 --dim=64 --epoch=8 --device=cuda:0 --gcnlayers=3 --lr=1e-3 --agg_type=mean --cdm_type=lightgcn --khop=1
```



## Inductive Scenario

```shell
python exp_ind.py  --exp_type=ind --method=icdm --datatype=EdNet-1 --test_size=0.2 --seed=0 --dim=64 --epoch=5 --device=cuda:0 --gcnlayers=3 --lr=2e-3 --agg_type=mean --mode=train --cdm_type=lightgcn --new_ratio=0.2 --khop=3
```

