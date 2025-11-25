# Intracranial Tumor Classification with the BRISC 2025 dataset
Deep learning project focused in utilizing transfer learning of ResNet for tumor classification from MRI data. The final model classifies 3 distinct types of tumors (Meningioma, Pituitary, Glioma) or the absence of one.

## Installation
**NOTE: TO USE CUDA, DO THE FOLLOWING BEFORE INSTALLING DEPENDENCIES**.  
Install CUDA Toolkit from: https://developer.nvidia.com/cuda-downloads  
You will need to separately install torch using these instructions: https://pytorch.org/get-started/locally/  

**AFTER YOU ARE DONE WITH CUDA INSTALLATION, OR YOU DON'T NEED CUDA**  
Required libraries can be installed from the requirements.txt using command:
```
pip install -r requirements.txt
```

Required project directory structure can be initialized by running the setup.py file with the root directory as the working directory.  
After initializing the structure, dataset should be manually saved in the data folder. args.py can be modified to fit your naming convention.

## Usage  
Most of the adjustable parameters and arguments reside in args.py. You should modify the parameters inside the file itself if you want to change anything.  
Noteworthy arguments for controlling training and evaluation:
```
-resume: boolean for resuming training from checkpoint
-train: boolean for toggling training of k-fold models
-evaluate: boolean for toggling model evaluation
```
These can be beneficial if your training loop terminates suddenly, or you want to only train or evaluate the model.

## Results
Results were achieved using transfer learning on Resnet34 with the default hyperparameters of args.py used.

### Validation
Using k-fold cross validation the following aggregated metrics were collected: (mean ± std)
```
Loss: 0.0498 ± 0.0165
Accuracy: 0.9890 ± 0.0038
Precision (macro): 0.9897 ± 0.0035
Recall (macro): 0.9889 ± 0.0039
F1 Score (macro): 0.9892 ± 0.0037
Cohen's Kappa: 0.9853 ± 0.0051
ROC-AUC (macro): 0.9991 ± 0.0005

Per-class F1: ['0.988', '0.982', '0.996', '0.991'] ± ['0.005', '0.007', '0.003', '0.003']
Per-class Precision: ['0.990', '0.988', '0.997', '0.984'] ± ['0.012', '0.010', '0.004', '0.006']
Per-class Recall: ['0.986', '0.977', '0.994', '0.999'] ± ['0.008', '0.015', '0.005', '0.002']
```
### Evaluation
Soft voting of k-fold ensemble was utilized for evaluation on the held-out test set. 
Metrics of ensemble evaluation:
```
Loss: 0.0237
Accuracy: 0.9947
Precision (macro): 0.9949
Recall (macro): 0.9947
F1 Score (macro): 0.9948
Cohen's Kappa: 0.9918
ROC-AUC (macro): 0.9998
Per-class F1: ['0.994', '0.990', '1.000', '0.995']
Per-class Precision: ['0.996', '0.987', '1.000', '0.997']
Per-class Recall: ['0.992', '0.993', '1.000', '0.993']
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
