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

## Dataset Attribution
This project uses the **BRISC 2025** dataset (https://www.kaggle.com/datasets/briscdataset/brisc2025), 
which is licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0).
© Amirreza Fateh, Yasin Rezvani, Sara Moayedi, Sadjad Rezvani, Fatemeh Fateh, Mansoor Fateh, Vahid Abolghasemi, 2025.
The dataset is not included in this repository.

## License
This project's **source code** is licensed under the MIT License.
The **dataset** is licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0) license.
