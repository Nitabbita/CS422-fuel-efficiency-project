# Predicting Car Fuel Efficiency Using k-Nearest Neighbors Regression

This project predicts a vehicle's fuel efficiency (Miles Per Gallon, MPG) using the Auto MPG dataset and a supervised machine learning model (k-Nearest Neighbors Regression). The work was completed as part of the CS 422 – Intro to Machine Learning course.

## 📦 Project Structure


```
project-root/
├── code/
│   └── knn_mpg.py
├── data/
│   └── auto-mpg.csv
├── report/
│   ├── fuel_efficiency_report.pdf
│   └── true_vs_predicted_mpg.png
├── slides/
│   └── presentation.pptx
└── README.md
```
## 📊 Dataset

Auto MPG dataset (UCI Machine Learning Repository)
The dataset contains vehicle specifications from 1970–1982 models.

Columns:
- mpg (target)
- cylinders
- displacement
- horsepower
- weight
- acceleration
- model year
- origin
- car name (removed for modeling)

Cleaning performed:
- Converted "?" in horsepower to numeric
- Removed rows with missing horsepower
- Final dataset size: 392 samples

## 🧠 Method

The goal is to predict fuel efficiency using basic vehicle attributes.

Approach:
1. Load cleaned dataset
2. Remove text feature (“car name”)
3. Split into training/test sets (80/20)
4. Standardize features
5. Train baseline model (predict mean MPG)
6. Search for optimal k in k-NN
7. Train final KNN model
8. Evaluate performance

Hyperparameter search:
k ∈ {1, 3, 5, 7, 9, 11, 13, 15}

## 🔥 Model Results

Baseline (mean) model:
- MAE: 5.881
- MSE: 51.620
- R²: -0.011

Best K: 3

kNN(k=3) model:
- MAE: 1.868
- MSE: 7.155
- R²: 0.860

Performance improved ~68% over baseline.

## 📈 Output Artifacts

A scatter plot of True MPG vs Predicted MPG is created automatically when you run the script:

report/true_vs_predicted_mpg.png

## 🖥️ How to Run

1. Install dependencies:
pip install pandas scikit-learn matplotlib numpy

2. Run the model from the project root:
python code/knn_mpg.py

The script will:
- Load the dataset
- Train baseline and KNN models
- Print metrics to the console
- Save plot to /report/

## 📄 Report & Presentation

- Full project report is located in /report/
- Slides are located in /slides/

Both include:
- Dataset description
- ML method explanation
- Experimental setup
- Results and discussion

## 🧰 Tools Used

- Python 3
- pandas
- scikit-learn
- numpy
- matplotlib

## 🔍 Notes

- No code chunks are included in the academic report, as required.
- The dataset is the standard cleaned Auto MPG dataset from UCI, widely used in ML examples.

## ⚖️ License

Academic use only. Dataset © UCI Machine Learning Repository.
