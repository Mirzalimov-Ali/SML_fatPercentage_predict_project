# Src Folder Haqida

`src/` folderi projectning **asosiy module va classlari** joylashgan bulib, malumotni yuklash, tozalash, feature yasash, model uqitish va log qilish jarayonlarini boshqaradi.  

Har bir modulening vazifasi va undagi asosiy methodlar yozilgan.

---

## 📂 Module va Classlar

### 1️⃣ `data_loader.py`
- **Vazifa:** Raw CSV filelarni yuklash, birlashtirish va saqlash.
- **Asosiy Class:** `DataLoader`
- **Metodelar:**
  - `load_and_merge()` – bir nechta CSV filelarni yuklab, bitta DataFrame ga birlashtiradi.
  - `save_merged(df)` – birlashtirilgan datasetni CSV filega saqlaydi.
- **Izoh:** Har bir file yuklanishi va birlashtirilishi logger orqali kuzatiladi, yoq file yoki xatolik bulsa log qilinadi.

---

### 2️⃣ `feature_engineering.py`
- **Vazifa:** Yangi featurelar yasash orqali model bashorat aniqligini oshirish.
- **Asosiy Class:** `FeatureCreation`
- **Metodlar:**
  - `create_features()` – quyidagi featurelarni yasaydi:
    1. `bmi` – vazn va buy asosida tana massasi indeksi.
    2. `hrr` – yurak urish rezervi (`max_bpm - resting_bpm`).
    3. `intensity` – mashq intensivligi HRR orqali.
    4. `calories_per_min` – mashq samaradorligi.
    5. `water_per_session` – mashq davomida suv istemoli.
    6. `experience_intensity_ratio` – tajriba darajasi bilan intensivlikni kupaytirish.
  - `getDataset()` – yangilangan DataFrame’ni qaytaradi.
- **Izoh:** Har bir feature yasash jarayoni logger orqali tekshiriladi.

---

### 3️⃣ `preprocessing.py`
- **Vazifa:** Datalarni modelga tayyorlash.
- **Asosiy Class:** `Preprocessing`
- **Metodlar:**
  - `fillMissingValues()` – numeric ustunlarni urtacha, categorical ustunlarni mode() bilan tuldiradi.
  - `encode()` – categorical ustunlarni label yoki one-hot encoding qiladi.
  - `scale()` – numeric ustunlarni 0-1 oraligiga keltiradi.
  - `logTransformation()` – skewed ustunlarga log transformationni qullaydi.
  - `getDataset()` – tayyorlangan datasetni qaytaradi.
- **Izoh:** Barcha methodlar chaining orqali ishlaydi, yani ketma-ket chaqirish mumkin.

---

### 4️⃣ `model_training.py`
- **Vazifa:** Modelni uqitish, embedded feature selection va baholash.
- **Asosiy Class:** `Trainer`
- **Metodlar:**
  - `train()` – train/test ajratadi, embedded feature selection orqali muhim featurelarni tanlaydi va modelni qayta uqitadi.
  - `evaluate()` – test setda bashorat qiladi, R2, MAE va K-Fold cross-validation natijalarini hisoblaydi.
- **Izoh:**  
  - Embedded feature selection tree-based modellar uchun `feature_importances_`, linear modellar uchun `coef_` ishlatadi.  
  - Tanlangan featurelar ustida model qayta uqitiladi.  
  - Har bir qadam logger orqali tekshiriladi.

---

### 5️⃣ `logger.py`
- **Vazifa:** Projectdagi barcha jarayonlarni log qilish.
- **Asosiy Funksiya:** `get_logger(file_name, log_file)`
  - Berilgan module uchun logger yasaydi.
  - Log darajalari: `INFO`, `ERROR`, `WARNING`.
  - Log filelar `logs/` folderga saqlanadi.
- **Izoh:** Har bir module bu logger orqali jarayonlarni kuzatadi va xatoliklarni aniqlash osonlashadi.

---

## ✅ Xulosa
`src/` folderi projectning **asosiy ishlov beruvchi modulelari**ni uz ichiga oladi.  
- **DataLoader** – datasetni yuklash va birlashtirish.  
- **FeatureCreation** – yangi featurelar yasash.  
- **Preprocessing** – data cleaning, encoding va scaling.  
- **Trainer** – model uqitish va embedded feature selection.  
- **Logger** – jarayonlarni kuzatish va debugging.  

Bu module structurening maqsadi — **har bir jarayonni module va methodlar bilan boshqarish** hamda modelni aniq va qayta ishlatiladigan qilib tayyorlash.