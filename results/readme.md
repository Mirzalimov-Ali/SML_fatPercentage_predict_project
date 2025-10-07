## 📊 Model Performance Comparison

`results/all_model_compare.txt` file barcha ishlatilgan algoritmlarning natijalarini solishtiradi. 
Bu fileda shu malumotlar mavjud:

- **Algorithm** – ishlatilgan model nomi  
- **r2_score** – modelning R² kursatkichi (qanchalik yaxshi fit qilgani)  
- **mean_absolute_error** – modelning urtacha xatolik qiymati  
- **K-Fold Mean** – K-Fold cross-validation orqali olingan urtacha natija  
- **K-Fold Std** – K-Fold natijalari tarqalishi

### Misol ko‘rinish:

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

> 🔹 Shu jadvaldan kurinib turibdiki, **DecisionTreeRegressor** eng yaxshi natijaga ega bulgan model hisoblanadi: eng yuqori R² va eng past MAE.  

