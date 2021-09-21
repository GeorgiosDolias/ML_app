# Regression Machine Learning models

## Demo app

Launch the app [![Open In Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/georgiosdolias/ml_app/main/ml-app.py)

## App info

This repository creates a Data Web App that enables users to interact with several hyper-parameters of Support Vector Regression algorithm from Sci-kit learn Python package. The created ML model is evaluated based on two performance metrics, namely: Coefficient of determination and mean squared error. The hyper-parameters are:


1. kernel
2. Regularization parameter C
3. Epsilon
4. Degree of polynomial kernel
5. Kernel coefficient gamma
6. Independent term
7. Hard limit on iterations
8. shrinking heuristic
9. size of the kernel cache
10. verbose output
11. Tolerance


## Default Dataset

Regarding the default dataset, the concrete slump test measures the consistency of fresh concrete before it sets. It is performed to check the workability of freshly made concrete, and therefore the ease with which concrete flows. It can also be used as an indicator of an improperly mixed batch.


## Reproducing the App

1. First, we create a virtual Python environment called my_venv
```
  python3 -m venv my_venv
```
2. Then, we activate the virtual environment
```
source path_to_your_virtual_environment/bin/activate
```
3. After getting to the virtual environment's file, install prerequisite packages
```
wget https://raw.githubusercontent.com/GeorgiosDolias/ML_app/main/requirements.txt
```
and
```
pip install -r requirements.txt
```
4. Dowload and unzip contents from Github repo

Dowload and unzip contents from https://github.com/GeorgiosDolias/ML_app/archive/main.zip

5. Launch the app
```
streamlit run ml-app.py
```


## Requirements

| Package | Version |
--- | ---
| streamlit | 0.88.0 |
| pandas |  1.1.3 |
| sci-kit learn | 0.23.2 |

## Useful Resources

1.  [Youtube tutorial from Chanin Nantasenamat (Data Professor) ](https://www.youtube.com/watch?v=8M20LyCZDOY )
2.  [Details of default dataset](https://archive.ics.uci.edu/ml/datasets/Concrete+Slump+Test)
