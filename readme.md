# Intracranial Tumor Classification with the BRISC 2025 dataset
Deep learning project focused in utilizing transfer learning of ResNet for tumor classification from MRI data. The final model classifies 3 distinct types of tumors (Meningioma, Pituitary, Glioma) or the absence of one.

## Installation
Required libraries can be installed from the requirements.txt using command:
```
pip install -r requirements.txt
```
**NOTE: TO USE CUDA, DO THE FOLLOWING**.  
Install CUDA Toolkit from: https://developer.nvidia.com/cuda-downloads  
You will need to separately install torch using these instructions: https://pytorch.org/get-started/locally/  

Required project directory structure can be initialized by running the setup.py file.  
After initializing the structure, dataset should be manually saved in the data folder.

## Usage  
Most of the changeable parameters and arguments reside in args.py. You should modify the parameters inside the file itself if you want to change anything.  
Noteworthy arguments for controlling training and evaluation:
```
-resume: boolean for resuming training from checkpoint
-train: boolean for toggling training of k-fold models
-evaluate: boolean for toggling model evaluation
```
These can be beneficial if your training loop terminates suddenly, or you want to only train or evaluate

## Results
Results were achieved using transfer learning on Resnet34.

### Validation
Using k-fold cross validation the following aggregated metrics were collected: (mean ± std)
```
  Loss: 0.0403 ± 0.0192
  Accuracy: 0.9880 ± 0.0032
  Precision (macro): 0.9884 ± 0.0024
  Recall (macro): 0.9880 ± 0.0033
  F1 Score (macro): 0.9881 ± 0.0029
  Cohen's Kappa: 0.9839 ± 0.0042
  ROC-AUC (macro): 0.9996 ± 0.0004
  
  Per-class F1: ['0.986', '0.982', '0.993', '0.991'] ± ['0.005', '0.004', '0.004', '0.007']
  Per-class Precision: ['0.991', '0.983', '0.992', '0.988'] ± ['0.008', '0.004', '0.008', '0.014']
  Per-class Recall: ['0.982', '0.980', '0.995', '0.995'] ± ['0.008', '0.008', '0.003', '0.002']
```
### Evaluation
Soft voting of k-fold ensemble was utilized for evaluation on the held-out test set. 
Metrics of ensemble evaluation:
```
  Loss: 0.0175
  Accuracy: 0.9956
  Precision: 0.9957
  Recall: 0.9956
  Macro F1 Score: 0.9956
  Cohen's Kappa: 0.9932
  ROC-AUC Macro: 0.9999
  Per-class F1: ['0.994', '0.992', '1.000', '0.997']
  Per-class Precision: ['0.996', '0.990', '1.000', '0.997']
  Per-class Recall: ['0.992', '0.993', '1.000', '0.997']
```
#### Confusion Matrix of ensemble evaluation:
![Confusion Matrix of Ensemble Predictions](results/confusion_matrix_ensemble.png)

## Dataset Attribution
This project uses the **BRISC 2025** dataset (https://www.kaggle.com/datasets/briscdataset/brisc2025), 
which is licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0).
© Amirreza Fateh, Yasin Rezvani, Sara Moayedi, Sadjad Rezvani, Fatemeh Fateh, Mansoor Fateh, Vahid Abolghasemi, 2025.
The dataset is not included in this repository.

## License
This project's **source code** is licensed under the MIT License.
The **dataset** is licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0) license.
