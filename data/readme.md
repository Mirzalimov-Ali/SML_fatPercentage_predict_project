# 📂 Data Folder Structure

Bu folderda projectda ishlatiladigan barcha ma’lumotlar saqlanadi.  
Datasetlar xom holatidan boshlab, tayyorlangan final versiyasigacha shu yerda bo‘ladi.

---

## 📁 [raw]

🗂 **Original dataset** (Kaggle yoki boshqa manbadan yuklab olingan).  
🚫 Hali ishlov berilmagan, toza xom fayllar shu yerda saqlanadi.

---

## 📁 [preprocessed]

✨ **Tozalangan datasetlar**:

-   Missing value’lar to‘ldirilgan
-   Encoding qilingan

---

## 📁 [engineered]

🛠 **Feature engineering qilingan datasetlar**:

-   Yangi feature’lar yaratilgan
-   Scaling qilingan
-   Log transformation ishlatilgan

> 🔎 Bu bosqichdagi datasetlar model sifatini oshirishga yordam beradi.

---

## 📁 [final]

🏁 **Tayyor datasetlar** – modelni trening va test qilishda ishlatiladigan **yakuniy fayllar** shu yerda bo‘ladi.
