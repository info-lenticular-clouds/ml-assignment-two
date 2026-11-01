# Landsat Land-Cover Classification

## a. Problem statement

A Landsat multispectral scanner records the same patch of ground in four spectral bands. From each 3×3 neighbourhood of pixels, the task is to predict the land-cover class of the centre pixel. The six classes are red soil, cotton crop, grey soil, damp grey soil, soil with vegetation stubble, and very damp grey soil.

This is a multi-class classification problem. Five models are trained on the official UCI training file and scored once on the official test file. The Streamlit app loads those saved models and scores an uploaded test CSV. It does not retrain them with the uploaded CSV file.

## b. Dataset description

**Source:** Statlog (Landsat Satellite), UCI Machine Learning Repository.  
https://archive.ics.uci.edu/dataset/146/statlog+landsat+satellite

The sample comes from a Landsat scene purchased from NASA by the Australian Centre for Remote Sensing. Ashwin Srinivasan converted a small labelled extract into the Statlog files. Class labels were assigned from a site visit. The UCI distribution is public.

| Item | Value |
| --- | --- |
| Instances | 6,435 (4,435 train, 2,000 test) |
| Features | 36 numeric spectral values, each in 0–255 |
| Target | `land_cover` (6 classes; UCI class code 6 does not occur) |
| Missing values | None |

Each row is one 3×3 neighbourhood. Pixels are stored left to right, top to bottom. Every pixel has four bands, in this order: green, red, near-infrared 1, near-infrared 2. Columns are named `{position}_{band}`, for example `center_red`. Attributes 17–20 in the raw file are the centre pixel. The label is the cover type of that centre pixel.

The official `sat.trn` / `sat.tst` split is used as published. The UCI documentation says not to replace that split with cross-validation, because overlapping neighbourhoods would leak spatial information into the test score. `test_data.csv` is the official test split with a header row.

Training-set class counts: red soil 1,072; very damp grey soil 1,038; grey soil 961; cotton crop 479; soil with vegetation stubble 470; damp grey soil 415. The test split has the same imbalance. Damp grey soil is the smallest class in both files.

## c. Github Repository Link

https://github.com/info-lenticular-clouds/ml-assignment-two

## d. Models used

All five models are fit on the same 4,435 training rows and scored on the same 2,000 test rows.

- **Logistic Regression.** Standardised features, L2 penalty, `C=1`, `class_weight="balanced"`.
- **Decision Tree.** Raw spectral values, `max_depth=16`, `min_samples_leaf=4`, `class_weight="balanced"`.
- **KNN.** Standardised features, `k=7`, Euclidean distance, neighbours weighted by inverse distance.
- **Naive Bayes.** Gaussian Naive Bayes on standardised features (`var_smoothing=1e-9`). Gaussian is the right variant because every feature is continuous.
- **Support Vector Machine.** Standardised features, RBF kernel, `C=10`, `gamma="scale"`, `class_weight="balanced"`.

Logistic regression, k-nearest neighbours, and the support-vector machine need standardised features because their geometry depends on feature scale but the tree does not. Naive Bayes is also given standardised features so each band contributes on a comparable scale.

Precision, recall, and F1 are weighted averages across the six classes. AUC is the weighted one-vs-rest area under the ROC curve. MCC is the multiclass Matthews correlation coefficient. For a single-label problem, weighted recall equals accuracy, which is why those two columns match. Precision and F1 still differ from accuracy.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.8435 | 0.9786 | 0.8597 | 0.8435 | 0.8492 | 0.8101 |
| Decision Tree | 0.8390 | 0.9372 | 0.8472 | 0.8390 | 0.8424 | 0.8032 |
| KNN | 0.9035 | 0.9841 | 0.9022 | 0.9035 | 0.9025 | 0.8814 |
| Naive Bayes | 0.7965 | 0.9589 | 0.8165 | 0.7965 | 0.8036 | 0.7531 |
| Support Vector Machine | 0.8945 | 0.9850 | 0.9037 | 0.8945 | 0.8970 | 0.8721 |

### Observations

| ML Model Name | Observation about model performance |
| --- | --- |
| Logistic Regression | A linear boundary is enough to rank the classes well (AUC 0.9786), and red soil and cotton crop are almost separated (F1 0.973 and 0.944). The grey-soil family is not linearly separable: damp grey soil falls to F1 0.571, with precision only 0.491. Overall accuracy is 0.8435. |
| Decision Tree | Accuracy (0.8390) is close to logistic regression, but AUC is the weakest of the five (0.9372) because a leaf gives a simplified, grouped, or less precise probability. The depth and leaf-size limits keep the tree from memorising neighbouring pixels i.e. overfitting. Damp grey soil is again the failure point (F1 0.552). |
| KNN | Best accuracy (0.9035), F1 (0.9025), and MCC (0.8814). Spectral classes form local clusters, so a distance-weighted vote of 7 neighbours matches the structure of this data. It also has the best F1 on the hard damp-grey class (0.700). Red soil reaches F1 0.985. |
| Naive Bayes | Lowest accuracy (0.7965) and MCC (0.7531). Gaussian Naive Bayes treats the 36 bands and neighbouring pixels as independent but actually they are not. The damage shows up on soil with vegetation stubble (F1 0.652, against 0.896 for KNN). AUC stays fairly high at 0.9589, so the scores still rank classes better than the hard predictions suggest. Cotton crop remains easy (F1 0.935). |
| Support Vector Machine | Closest model to KNN on the actual decisions (accuracy 0.8945, F1 0.8970, MCC 0.8721) and the best ranking score (AUC 0.9850). Balanced class weights lift damp-grey recall to 0.782, above KNN’s 0.673, but precision drops to 0.609 because the model over-calls that class. |
| Overall, Winner for your dataset? | **KNN.** It leads on accuracy, F1, and MCC, and it is the most balanced model on damp grey soil, which is the class every model struggles with. The RBF support-vector machine is a close second and has the highest AUC. Logistic regression and the decision tree are clearly behind those two. Gaussian Naive Bayes is last because the independence assumption does not hold good for a 3×3 multispectral neighbourhood. |

## How to run

From this folder, with the packages in `requirements.txt` installed:

```bash
python train_models.py
streamlit run app.py
```

Training done on `data/sat.trn` and evaluated on `data/sat.tst` which is copied to `test_data.csv` on root directory, and saves the five models under `models/` directory. The app scores the bundled test file immediately. Use the sidebar to switch between models or upload another test CSV with the same feature columns and a `land_cover` as target column.
