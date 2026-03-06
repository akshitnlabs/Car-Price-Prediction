# 🚗 Car Price Prediction — Linear Regression

> Building a multiple linear regression model to identify the key variables that drive automobile pricing in the US market.

---

## 🧠 Business Problem

A Chinese automobile company **Geely Auto** is planning to enter the US market and wants to understand what factors significantly influence car pricing in America — as these may differ greatly from the Chinese market.

The goal is to answer two questions:
1. Which variables are significant in predicting the price of a car?
2. How well do those variables describe the price of a car?

---

## 📁 Dataset

| File | Description |
|------|-------------|
| `CarPrice_Assignment.csv` | 205 car records with specifications including engine type, dimensions, fuel type, company, and price |

**Target Variable:** `price` — the selling price of the car in the US market

**Key Features:** engine size, curb weight, horsepower, car width/length, fuel type, aspiration, engine location, cylinder count, car company, and more.

---

## 🔍 Analysis Workflow

### 1. Data Understanding & Cleaning
- Parsed `CarName` column into `CarCompany` and `CarModel` using string splitting
- Fixed inconsistent company name spellings: `vokswagen` → `volkswagen`, `toyouta` → `toyota`, `maxda` → `mazda`, `porcshce` → `porsche`, etc.
- Dropped `CarModel` and `car_ID` as non-predictive columns

### 2. Exploratory Data Analysis
- **Pair plot** for all numeric variables to identify linear relationships with price
- **Box plots** for categorical variables (fuel type, aspiration, drive wheel, engine location, carbody, engine type, car company) against price
- **Heatmap** to identify multicollinearity among features

### 3. Data Preparation
- Created **dummy variables** for all categorical columns using `pd.get_dummies()`
- Applied **MinMaxScaler** to normalize numeric features to the same scale
- **70/30 train-test split** with `random_state=100` for reproducibility

### 4. Model Building (Iterative Approach)
Built the model incrementally, adding variables one at a time and monitoring R² improvement:

| Step | Variables Added | Key Observation |
|------|----------------|-----------------|
| 1 | `enginesize` | R² = 0.75 — good start |
| 2 | + `curbweight` | Adjusted R² improved |
| 3 | + `horsepower` | Further improvement |
| 4 | + `carwidth` | Change observed in p-values |
| 5 | + `carlength` | Continued evaluation |
| 6 | All variables | Multicollinearity warning detected |

### 5. Feature Selection — VIF & p-value
- Computed **Variance Inflation Factor (VIF)** for all features
- Iteratively dropped high-VIF and high-p-value variables to eliminate multicollinearity
- **Final model variables:** `enginesize`, `aspiration_turbo`, `enginelocation_rear`, `enginetype_rotor`, `cylindernumber_four`, `cylindernumber_six`

### 6. Residual Analysis
- Plotted distribution of error terms to verify **normality of residuals**
- Confirmed errors are approximately normally distributed — a key OLS assumption

### 7. Model Evaluation
- Scatter plot of actual vs. predicted prices on the test set
- Evaluated final model using **R² score on the test set**

---

## 📐 Final Model Equation

```
Price = (enginesize × 1.0672)
      + (aspiration_turbo × 0.0670)
      + (enginelocation_rear × 0.2846)
      + (enginetype_rotor × 0.1294)
      + (cylindernumber_four × -0.1216)
      + (cylindernumber_six × -0.0641)
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3 | Core language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Matplotlib & Seaborn | Visualization |
| Scikit-learn | Train-test split, MinMaxScaler, R² score |
| Statsmodels (OLS) | Linear regression modeling, p-values, VIF |
| Jupyter Notebook | Interactive development |

---

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/akshitnlabs/<repo-name>.git
cd <repo-name>

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels jupyter

# Launch the notebook
jupyter notebook Linear_Regression_Assignment.ipynb
```

> **Note:** Place `CarPrice_Assignment.csv` in the same directory as the notebook before running.

---

## 💡 Key Findings

- **Engine size** is the single strongest predictor of car price (R² = 0.75 with engine size alone)
- **Turbo aspiration**, **rear engine location**, and **rotary engine type** are associated with significantly higher prices — premium features that buyers pay more for
- **Four and six cylinder** engines are associated with lower prices compared to the baseline
- Many physical dimensions (car width, length, curb weight) show high multicollinearity and were excluded from the final model to improve reliability
- The final model achieves strong predictive performance on the test set with well-distributed residuals

---

## 👤 Author

**Akshit Nair** — [GitHub](https://github.com/akshitnlabs)

---

*This project was completed as part of a machine learning assignment to develop hands-on skills in linear regression, feature selection, and model evaluation.*
