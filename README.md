# 🏋️ Gym Dataset Fat Prediction Project

## Project haqida 
Bu project **odamlarning fitness malumotlariga asoslanib bodyfat (%) bashorat** qiladi. 
Project **regression** turiga kiradi, yani maqsadimiz **sonli (continuous) qiymatni taxmin qilish**.

## 🎯 Target
- Target ustuni: `fat_percentage`  
- Maqsad: odamning tana yog foizini turli featurelar (age, weight, height, exercise, heart rate va boshqalar) orqali aniqlash.

🔹 **Future Vision**
- Dataset
- Yangi Featurelar yaratish
- Data Analysis & Visualization
- Modellar va natijalar
- Xulosa
- O‘rnatish
- License

---

📊 **Dataset**
Datasetda sport zaliga boruvchi odamlar buyicha malumotlar mavjud va quyidagi ustunlarni uz ichiga oladi:

| Columns              | Type    | Description                                           |
|----------------------|---------|-------------------------------------------------------|      
| age                  | int     | Foydalanuvchi yoshi                                   |
| gender               | object  | Foydalanuvchi jinsi                                   |
| weight               | float   | Og‘irligi (kg)                                        |
| height               | float   | Bo‘yi (cm)                                            |
| max_bpm              | int     | Maksimal yurak urishi                                 |
| avg_bpm              | int     | O‘rtacha yurak urishi                                 |
| resting_bpm          | int     | Dam olishdagi yurak urishi                            |
| session_duration     | float   | Mashg‘ulot davomiyligi (minut)                        |
| calories_burned      | float   | Mashg‘ulot davomida sarflangan kaloriyalar            |
| workout_type         | object  | Mashg‘ulot turi                                       |
| fat_percentage       | float   | Tana yog‘ foizi                                       |
| water_intake         | float   | Suv ichish miqdori (litr)                             |
| workout_frequency    | int     | Haftalik mashg‘ulotlar soni                           |
| experience_level     | object  | Mashg‘ulot tajribasi (beginner/intermediate/advanced) |
| bmi                  | float   | BMI (Body Mass Index)                                 |
| age_category         | object  | Yoshlarga qarab kategoriyalash                        |
| weight_category      | object  | Og‘irlik bo‘yicha kategoriyalash                      |
| bmi_category         | object  | BMI bo‘yicha kategoriyalash                           |

---

🔹 **Yangi Featurelar**
Project davomida quyidagi yangi featurelar yaratildi, ular model aniqligini oshirishga yordam beradi:

| Feature                    | Tavsif                                                   |
|----------------------------|---------------------------------------------------------|
| experience_intensity_ratio | Tajriba darajasi va mashg‘ulot davomiyligi nisbati     |
| session_duration_ratio     | Mashg‘ulot davomiyligi va kunlik optimal vaqt nisbati  |
| age_category               | Yoshlarga qarab guruhlash                              |
| weight_category            | Og‘irlik bo‘yicha kategoriyalash                       |
| bmi_category               | BMI bo‘yicha kategoriyalash                            |

---

📈 **Visualization (Data Analysis)**  
Barcha grafiklar `visuals/` papkada saqlandi:

- Age vs Fat Percentage
- Weight vs Fat Percentage
- Workout Frequency vs Calories Burned
- Gender vs Fat Percentage
- BMI Distribution
- Heart Rate Metrics (Max, Avg, Resting)
- Session Duration vs Calories Burned
- Experience Level Impact
- Interaktiv scatter plot (plotly-express)

---

🤖 **Modellar va natijalar**

```                                     
## Model Compare   
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

🔹 **Xulosa**
- Fat percentage ga eng katta tasir qiluvchi faktorlar: **weight, BMI, session duration, workout frequency**.  
- Yangi featurelar (experience_intensity_ratio, session_duration_ratio, bmi_category) model aniqligini oshirdi.  
- Eng yaxshi model: **DecisionTreeRegressor** (R2 ≈ 0.91, MAE ≈ 0.12).

---

⚙️ **O‘rnatish**

```bash
git clone https://github.com/username/Gym_Fat_Percentage_Prediction.git
cd Gym_Fat_Percentage_Prediction
pip install -r requirements.txt


