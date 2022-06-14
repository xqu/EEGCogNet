# Introduction EEGCogNet
EEGCogNet Dataset is a benchmark to evaluate cognitive tasks based on EEG signals from non-expert college students. **We need to add more details**

# Overview
**We need to add a description for Overview**

# Installation
We highly recommend using a conda environment for package management.

First, you should create a conda environment
```
conda create -n EEGCogNet python = 3.9
```

You should download the sklearn, pandas, and matplotlib:
```
conda install scikit-learn
```
```
conda install pandas
```
```
conda install matplotlib
```

# Evaluating the Benchmark and Basline
First, you should create these four directories: output/Python_Math_2018, output/Raw_2018, output/rwt_old_results, and output/tcr_old_results so that the results for Python_Math, GRE_Relax, RWT, and TCR will be generated in these directories, respectively.

Then, activate the EEGCogNet conda environment:
```
conda activate EEGCogNet
```

## Python_Math
```
python Python_Math.py
```

## GRE_Relax
```
python GRE_Relax.py
```

## RWT
```
python RWT.py
```

## TCR
```
python TCR.py
```




