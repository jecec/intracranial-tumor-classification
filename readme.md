## Intracranial Tumor Classification with the BRISC 2025 dataset
Deep learning project focused in utilizing transfer learning for tumor classification from MRI data.

## Installation
Required libraries can be installed from the requirements.txt using command:
```
pip install -r requirements.txt
```
**NOTE: THIS DOES NOT INSTALL TORCH OR TORCHVISION**.  
You will need to separately install torch using these instructions if you desire to use CUDA: https://pytorch.org/get-started/locally/  
If not, you can install torch and torchvision normally with:
```
pip install torch torchvision
```
Required project directory structure can be initialized by running the setup.py file.  
After initializing the structure, dataset should be saved in the data folder.

## Usage  
Most of the changeable parameters and arguments reside in args.py. You should modify the parameters inside the file itself if you want to change anything.  
Noteworthy arguments for controlling training and evaluation:
```
-resume: boolean for resuming training from checkpoint
-train_cv: boolean for controlling if k-fold cross validation should be used and trained
-train_main: boolean for controlling if the final singular model should be trained
-evaluate: boolean for controlling if the final singular model should be evaluated
```
These can be beneficial if you are troubleshooting specific phases or want to focus only on k-fold-cv training for example.

## Results
Results were achieved using transfer learning on Resnet18.

Using K-Fold Cross Validation the following aggregated metrics were collected: (mean ± std)
```
Loss: 0.0386 ± 0.0106
Accuracy: 0.9892 ± 0.0016
Balanced Accuracy: 0.9892 ± 0.0016
Precision (macro): 0.9898 ± 0.0012
Recall (macro): 0.9892 ± 0.0016
F1 Score (macro): 0.9894 ± 0.0014
Cohen's Kappa: 0.9858 ± 0.0023
ROC-AUC (macro): 0.9995 ± 0.0005

# Classes in order: [glioma, meningioma, no_tumor, pituitary]
Per-class F1: ['0.986', '0.985', '0.994', '0.993'] ± ['0.002', '0.005', '0.004', '0.005']
Per-class Precision: ['0.991', '0.979', '0.995', '0.993'] ± ['0.006', '0.009', '0.007', '0.006']
Per-class Recall: ['0.981', '0.990', '0.993', '0.992'] ± ['0.008', '0.005', '0.009', '0.004']
```
Trained a singular model on the entire training set after achieving satisfactory results with K-Fold Cross Validation.  
Confusion Matrix of final model evaluation on the held out test set:
![Confusion Matrix of Final Model](results/confusion_matrix_main.png)

## Dataset Attribution
This project uses the **BRISC 2025** dataset (https://www.kaggle.com/datasets/briscdataset/brisc2025), 
which is licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0).
© Amirreza Fateh, Yasin Rezvani, Sara Moayedi, Sadjad Rezvani, Fatemeh Fateh, Mansoor Fateh, Vahid Abolghasemi, 2025.
The dataset is not included in this repository.

## License
This project's **source code** is licensed under the MIT License.
The **dataset** is licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0) license.
