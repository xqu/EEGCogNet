# Introduction EEGCogNet
Welcome to the GitHub Repository of EEGCogNet, a dataset of Electroencephalography (EEG) recordings collected with affordable technology from 120 non-expert subjects while performing frequent and common cognitive activities. 

## Overview
The dataset is the aggregation of the results from four separate experiments: Read-Write-Type (RWT), Think-Count-Recall (TCR), GRE-Relax (GRE), Math-Shut-Read-Open (MSR). The repository provides the general functionality to run the benchmark on our four datasets, including both standard machine learning algorithms and deep learning neural networks. We support the following methods:

### Machine learning:
```
Gradient Boosting
LDA
Nearest Neighbors
Ada Boost
Random forest
Linear SVM
RBF SVM
Decision Tree
Shrinkage LDA
```
### Deep Learning:
```
Convolutional Neural Network
Long Short-term Memory 
Attention-based Transformer
```
Kindly note that the hyperparameters for Transformer has not been tuned and it doesn't yield convincing results at the moment.


## Installation
EEGCogNet requires the following packages: 

The following libraries are needed to run this benchmark. We are using the newsest version of these packages as of June 6th, 2022. 

Data processing and visualization:
pandas
matplotlib
numpy

Machine learning:
sklearn

Deep learning: 
pytorch

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

In addition, you should download PyTorch if you want to run the deep learning algorithms. 

For Windows and Linux, do:
```
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

For Mac:
```
conda install pytorch torchvision torchaudio -c pytorch
```
Note the stable version of PyTorch (1.11.0) does not support M1 Mac. Please download the Preview version, which is also compatible with the EEGCogNet codebase. 

## Evaluating the Benchmark and Baseline
First, you should create these four directories: output/Raw_Python_Math, output/GRE_Raw, output/rwt_old_results, and output/tcr_old_results so that the results for Python_Math, GRE_Relax, RWT, and TCR will be generated in these directories, respectively.

Then, activate the EEGCogNet conda environment:
```
conda activate EEGCogNet
```

If module not found error is prompted, make sure to check if the required packages are installed in the EEGCogNet virtual environment. You can do this by:
```
conda list
```

Once in the derictory that contains the four python files that correspond to each task, type the following command in your terminal based on which dataset you want to evaluate.

### Python_Math
```
python Python_Math.py
```

### GRE_Relax
```
python GRE_Relax.py
```

### RWT
```
python RWT.py
```

### TCR
```
python TCR.py
```

## Deep Learning
By default, neural network based deep learning methods are excluded when running the files above. To include deep learing models, simply change the variable 
```
INCLUDE_DL = False
```
to true. You can also adjust the batch size and the number of CPU cores. By default, the deep learnintg models will automatically run on GPU if CUDA is available.



