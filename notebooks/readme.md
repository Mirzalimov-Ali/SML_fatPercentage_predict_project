# 📒 Notebooks Folder Overview

`notebooks/` folderi projectning **bosqichma-bosqich malumotlarni urganish, tozalash, feature engineering va model tayyorlash jarayonlarini** Jupyter Notebook kurinishida jamlagan.
Har bir notebook uz vazifasiga ega va bir-birini tuldiradi: malumotni urganamiz → tozalaymiz → yangi featurelar yasaymiz → modellaymiz → foydalanuvchi uchun bashorat qilamiz.

---

## 📁 Notebooklar tuzilishi

* `data_analysis.ipynb`
* `preprocessing.ipynb`
* `feature_engineering.ipynb`
* `model_training.ipynb`

---

## 📊 1. data_analysis.ipynb

🔎 **Exploratory Data Analysis (EDA)**
Ushbu notebookda dataset ustida **vizualization va tahlil** ishlari amalga oshirilgan:

* Seaborn va Matplotlib yordamida:

  * Lineplot: Average Calories Burned by Workout Frequency
  * Boxplot: Calories Burned by BMI Category
  * Histogram: Distribution of Calories Burned
  * Scatterplot: Calories Burned vs Experience Intensity Ratio
  * Scatterplot: Fat Percentage vs Age
  * Barplot: Average Calories Burned by Experience Level
  * Pairplot: Numeric features bilan fat_percentage o‘rtasidagi bog‘lanish
  * Heatmap: Features correlation
  * Violinplot: Calories Burned by Workout Frequency va Gender

* Plotly Express orqali **interactive bubble plot**: Weight vs Fat Percentage, session duration va experience intensity ratio bilan vizualization

* Grafiklar saqlangan folder: `visuals/`

---

## ⚙️ 2. preprocessing.ipynb

🧹 **Datasetni tozalash va oldindan tayyorlash**

* `Preprocessing` class yordamida:

  * Missing valuelarni tuldirish
  * Categorical columnlarni `LabelEncoder` bilan codelash
  * `MinMaxScaler` bilan scaling
* Natijada **preprocessed dataset** tayyorlanadi:
  📂 `data/preprocessed/preprocessed_gym_dataset.csv`

---

## 🛠 3. feature_engineering.ipynb

⚡ **Yangi feature’lar yaratish**

* `FeatureCreation` class yordamida quyidagi featurelar qushilgan:

  * Experience Intensity Ratio
  * Session Duration ratio
  * Age Category
  * Weight Category
  * BMI Category
* Natijada yasalgan dataset saqlanadi:
  📂 `data/engineered/engineered_gym_dataset.csv`
  📂 `data/final/final_gym_dataset.csv`

---

## 🏋️ 4. model_training.ipynb

⚡ **Nima qilindi:**  
* Final dataset import qilindi va tekshirildi  
* Features va target ajratildi, train/test split qilindi  
* Har bir model R2, MAE, K-Fold bilan baholandi  
* Eng yaxshi model tanlandi va saqlandi  
* Natijalar jadval va rangli console bilan vizual qilindi  
* Barcha jarayonlar logger bilan yozib borildi

### 🏗 Modellar

Quyidagi regression modellari solishtirilgan:

* LinearRegression
* LassoCV
* RidgeCV
* ElasticNetCV
* DecisionTreeRegressor
* RandomForestRegressor
* GradientBoostingRegressor
* ExtraTreesRegressor
* HistGradientBoostingRegressor
* AdaBoostRegressor
* XGBRegressor
* SVR
* KNeighborsRegressor

### 📊 Baholash mezonlari

* r2_score
* mean_absolute_error
* cross_val_score (K-Fold mean va std)

### 🏆 Eng yaxshi model

* **HistGradientBoostingRegressor**
* Model `joblib` orqali saqlanadi:
  📂 `model/HistGradientBoostingRegressor.joblib`

---

## 📌 Umumiy Jarayon

1. `preprocessing.ipynb`        → raw datasetni tozalash
2. `feature_engineering.ipynb`  → yangi feature’lar yaratish
3. `data_analysis.ipynb`        → vizualization va tahlil
4. `model_training.ipynb`       → ML modellari bilan trening va comparison

🚀 Shu tartibda project **toza pipeline** ko‘rinishida tashkil etilgan.
