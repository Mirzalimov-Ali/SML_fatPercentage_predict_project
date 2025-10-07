## 🧾 Logs 

Projectda har bir bosqich uchun alohida log filelar yasalgan. Bu loglar datalar oqimini kuzatish, xatoliklarni aniqlash va jarayonlarni tahlil qilishda yordam beradi. 

---

### 📂 `data_loader.log`

Raw datasetni (`gym_dataset1.csv`, `gym_dataset2.csv`) yuklab olish, merge qilish va yakuniy `gym_dataset.csv` fileni yasash jarayonlarini qayd etadi.

**Example:**
```
2025-10-04 13:10:35,700 - INFO - data_loader - Loaded file: data/raw\gym_dataset1.csv
2025-10-04 13:10:35,708 - INFO - data_loader - Data merged successfully. Shape: (2644, 18)
2025-10-04 13:10:35,731 - INFO - data_loader - Merged data saved to data/raw\gym_dataset.csv
```

**Key Highlights:**

* Har bir file yuklangani loglanadi
* Birlashtirilgan datasetning ulchami kursatiladi
* Yangi CSV file saqlangan manzil qayd etiladi

---

### ⚙️ `preprocessing.log`

Fill missing values, scaling va transformation bosqichlarini kuzatadi.

**Example:**

```
2025-10-04 13:30:53,298 - INFO - use_preprocessing - Dataset loaded with shape: (2644, 18)
2025-10-04 13:30:53,311 - INFO - use_preprocessing - Data success preprocessed!
2025-10-04 13:30:53,311 - INFO - use_preprocessing - File created to path: data/preprocessed\preprocessed_gym_dataset.csv
```

**Key Highlights:**

* Dataset yuklanishi va preprocessed holatiga utganligi loglanadi
* Yangi file nomi va joylashuvi aniq yoziladi

---

### 🧩 `feature_engineering.log`

Feature yasash (bmi, hrr, calories_per_min va boshqalar) hamda datasetni scaling va transforation qilish jarayonlarini qayd etadi.

**Example:**

```
2025-10-04 13:49:10,341 - INFO - feature_creation - Feature 'bmi' created
2025-10-04 13:49:10,342 - INFO - feature_creation - Feature 'hrr' created
2025-10-04 13:49:10,399 - INFO - use_feature_engineering - Success preprocessed! (Scaled, transformated)
```

**Key Highlights:**

* Yasalgan har bir feature alohida loglanadi
* Scaling va transformation muvaffaqiyatli yakunlanganligi qayd etiladi
* Yakuniy dataset `data/final/final_gym_dataset.csv` sifatida saqlanadi

---

### 🤖 `model_training.log`

Model training, baholash va natijalarni taqqoslash jarayonini yozadi. Har bir regressorni baholash natijasi loglanadi.

**Example:**

```
2025-10-07 09:12:55,460 - INFO - training - Model KNeighborsRegressor predicted successfully
2025-10-07 09:12:55,513 - INFO - training - Evaluation done for model KNeighborsRegressor: r2=0.634, mae=3.155, kfold_mean=0.623
2025-10-07 09:12:55,524 - INFO - use_train - Comparison table saved at results/all_model_compare.txt

+-------------------------------+-------+-------+---------------+--------------+
| Algorithms                    |    R2 |   MAE |   K-Fold Mean |   K-Fold Std |
+===============================+=======+=======+===============+==============+
| DecisionTreeRegressor         | 0.808 | 2.402 |         0.808 |        0.023 | ==> Best Model
+-------------------------------+-------+-------+---------------+--------------+
| AdaBoostRegressor             | 0.798 | 2.405 |         0.806 |        0.021 |
+-------------------------------+-------+-------+---------------+--------------+
| GradientBoostingRegressor     | 0.797 | 2.418 |         0.798 |        0.021 |
+-------------------------------+-------+-------+---------------+--------------+
| ExtraTreesRegressor           | 0.784 | 2.455 |         0.772 |        0.029 |
+-------------------------------+-------+-------+---------------+--------------+
| HistGradientBoostingRegressor | 0.776 | 2.474 |         0.770 |        0.029 |
+-------------------------------+-------+-------+---------------+--------------+
| RandomForestRegressor         | 0.756 | 2.546 |         0.750 |        0.028 |
+-------------------------------+-------+-------+---------------+--------------+
| XGBRegressor                  | 0.738 | 2.610 |         0.749 |        0.021 |
+-------------------------------+-------+-------+---------------+--------------+
| SVR                           | 0.716 | 2.873 |         0.702 |        0.035 |
+-------------------------------+-------+-------+---------------+--------------+
| KNeighborsRegressor           | 0.634 | 3.156 |         0.623 |        0.053 |
+-------------------------------+-------+-------+---------------+--------------+
| LinearRegression              | 0.633 | 3.201 |         0.627 |        0.034 |
+-------------------------------+-------+-------+---------------+--------------+
| ElasticNetCV                  | 0.632 | 3.188 |         0.631 |        0.035 |
+-------------------------------+-------+-------+---------------+--------------+
| RidgeCV                       | 0.632 | 3.181 |         0.640 |        0.025 |
+-------------------------------+-------+-------+---------------+--------------+
| LassoCV                       | 0.631 | 3.206 |         0.635 |        0.035 | ==> Worst Model
+-------------------------------+-------+-------+---------------+--------------+
```

**Key Highlights:**

* Model nomi, baholash metrics (R2, MAE, K-Fold Mean/Std) loglanadi
* Natijalar jadvali `results/all_model_compare.txt` filega saqlanadi
* Eng yaxshi modelni aniqlash jarayonini kuzatish osonlashadi

---
