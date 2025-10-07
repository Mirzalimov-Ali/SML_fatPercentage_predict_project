# Scripts Papkasi Haqida

`scripts/` folderi project jarayonlarini **amaliy bajaruvchi scriptlari**ni uz ichiga oladi.  
Ular `src/` folderidagi modulelarni chaqiradi va datani **yuklash, preprocessing, feature engineering va model training** bosqichlarini ketma-ket bajaradi.  

Har bir script **logger** orqali jarayonlarni qayd qiladi va natijalarni saqlaydi.

---

## 📂 Scriptlar va ularning vazifalari

### 1️⃣ `model_training.py`
- **Vazifa:** Turli regression algorithmlarini train qilish, ularni baholash va eng yaxshi modelni saqlash.
- **Jarayonlar:**
  1. Datasetni `final_gym_dataset.csv` dan yuklash (`pandas` yordamida).
  2. Regression algoritmlarini yasash:
     - **LinearRegression**
     - **LassoCV**
     - **RidgeCV**
     - **ElasticNetCV**
     - **DecisionTreeRegressor**
     - **RandomForestRegressor**
     - **GradientBoostingRegressor**
     - **HistGradientBoostingRegressor**
     - **ExtraTreesRegressor**
     - **AdaBoostRegressor**
     - **XGBRegressor**
     - **SVR** (RBF kernel)
     - **KNeighborsRegressor**
  3. Har bir modelni `Trainer` classi yordamida:
     - `train()` – train/test split va embedded feature selection bilan train qilish.
     - `evaluate()` – test setda R2, MAE, K-Fold cross-validation natijalarini hisoblash.
  4. Eng yaxshi modelni aniqlash va `model/best/` folderiga saqlash.
  5. Barcha modellar natijalarini taqqoslash va `results/all_model_compare.txt` ga saqlash.
- **Ishlatilgan methodlar va kontseptsiyalar:**
  - `Trainer` classi (train + evaluate)
  - Embedded feature selection (`feature_importances_` yoki `coef_` orqali)
  - Logger orqali barcha qadamlar tekshiriladi.
  - Natijalarni chiroyli table kurinishida chiqarish (`rich` va `tabulate`).

```

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


---

### 2️⃣ `use_data_loader.py`
- **Vazifa:** Raw CSV filelarni birlashtirish va saqlash.
- **Jarayonlar:**
  - `DataLoader` classi chaqiriladi.
  - `load_and_merge()` yordamida `gym_dataset1.csv` va `gym_dataset2.csv` filarlarni birlashtiriladi.
  - `save_merged()` orqali `gym_dataset.csv` sifatida saqlanadi.
- **Izoh:** Logger orqali file yuklash, birlashtirish va saqlash jarayoni kuzatiladi.

---

### 3️⃣ `use_feature_engineering.py`
- **Vazifa:** Datasetga yangi featurelar qushish va keyingi preprocessingni bajarish.
- **Jarayonlar:**
  1. `FeatureCreation` classi orqali yangi featurelar yasash:
     - `bmi`, `hrr`, `intensity`, `calories_per_min`, `water_per_session`, `experience_intensity_ratio`
  2. Preprocessed datasetni scale qilish va log transformation qilish (`Preprocessing` classi yordamida).
  3. Yakuniy datasetni `final_gym_dataset.csv` sifatida saqlash (`data/final/` folderga).
- **Izoh:**  
  - Feature engineering va preprocessing bir scriptda ketma-ket amalga oshiriladi.  
  - Logger barcha qadamlarni qayd qiladi.

---

### 4️⃣ `use_preprocessing.py`
- **Vazifa:** Raw datasetni tozalash va preprocessed formatga tayyorlash.
- **Jarayonlar:**
  1. `Preprocessing` classidan foydalanish.
  2. `fillMissingValues()` – numerical va categorical ustunlarni tuldirish.
  3. `encode()` – categorical ustunlarni label yoki one-hot encoding orqali transformation qilish.
  4. Tayyorlangan datasetni `preprocessed_gym_dataset.csv` sifatida saqlash (`data/preprocessed/`).
- **Izoh:**  
  - Script faqat preprocessingga etibor qaratadi.  
  - Feature creation va scaling keyingi scriptlarda bajariladi.  
  - Logger orqali jarayonlar kuzatiladi.

---

## ⚙️ Jarayon mantigi

1. **Data yuklash:** `use_data_loader.py`  
   - Raw CSV filelar birlashtiriladi va yagona datasetga aylanadi.
2. **Preprocessing:** `use_preprocessing.py`  
   - Missing values tuldiriladi, categorical ustunlar encode qilinadi.  
   - Preprocessed dataset saqlanadi.
3. **Feature Engineering:** `use_feature_engineering.py`  
   - Yangi featurelar qushiladi, scale va log transformation qilinadi.  
   - Yakuniy dataset saqlanadi (`final_gym_dataset.csv`).
4. **Model Training:** `model_training.py`  
   - Turli regressiya algoritmlari o‘qitiladi va baholanadi.  
   - Embedded feature selection orqali muhim feature’lar tanlanadi.  
   - Eng yaxshi model saqlanadi, barcha modellar taqqoslanadi.

---

## ✅ Xulosa

`scripts/` folderi project jarayonlarini **amaliy bajaruvchi skriptlar**ni uz ichiga oladi:  

- Datani yuklash va birlashtirish  
- Preprocessing (tozalash, encode qilish)  
- Feature engineering va transformation  
- Model training va evaluation  
- Eng yaxshi modelni saqlash va natijalarni taqqoslash  

**Xususiyatlar:**
- Modular va ketma-ket ishlash imkoniyati mavjud.  
- Har bir qadam **logger** orqali kuzatiladi.  
- Embedded feature selection va K-Fold cross-validation bilan model baholash amalga oshiriladi.  
- Foydalanuvchi uchun barcha intermediate va yakuniy datasetlar saqlanadi.
