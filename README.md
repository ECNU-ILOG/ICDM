# Inductive Cognitive Diagnosis Model (ICDM)
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

