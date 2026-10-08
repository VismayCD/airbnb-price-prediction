# Predicting Airbnb Listing Price Across Cities (NYC, Paris, Berlin)

**Course:** UE24CS352A – Machine Learning (Mini-Project)
**Team:** Vismay Chandra Dev (PES2UG24CS597), Vishwas Hosamath (PES2UG24CS596) – Section 5J

We predict nightly Airbnb listing prices with Ridge, KNN, Random Forest, XGBoost and two
neural networks, and test whether a neural network trained on some cities transfers to an
unseen one. The task follows *Predicting Airbnb Listing Price Across Different Cities*
(Luo, Zhou, Zhou, 2019); this is an independent re-implementation with additional
transfer-learning diagnostics.

## Key results (NYC unless stated; full tables in `docs/writeup.pdf`)
| Experiment | Result |
|---|---|
| Best model (continuous features, raw price, ≤ $500) | XGBoost, test R² 0.740 (RF 0.717, NN 0.666/0.686, KNN 0.607, Ridge 0.446) |
| Log-price label (uncapped data) | R² 0.470 → 0.808 |
| All features + log label | NYC 0.825, Paris 0.826 |
| Transfer to unseen Berlin | ranking transfers (Spearman 0.79–0.88), price level does not; 100 local labels recover R² to 0.71–0.84 |

## 1. Setup
```bash
git clone https://github.com/<user>/<repo>.git
cd <repo>
python -m venv venv
venv\Scripts\activate          # Windows   (Linux/macOS: source venv/bin/activate)
pip install -r requirements.txt
```

## 2. Data (not committed – place the three CSVs in `data/`)
Use the **detailed** Inside Airbnb listing tables (`listings.csv.gz`, ~90–96 columns), unzipped and
renamed: `data/nyc.csv`, `data/paris.csv`, `data/berlin.csv`.
Source: <http://insideairbnb.com/get-the-data.html> (the paper used Kaggle mirrors of the same data).
Audit a file before running:
```bash
python check_data.py data/nyc.csv data/paris.csv data/berlin.csv
```

## 3. Run
```bash
python airbnb_price.py --nyc data/nyc.csv --paris data/paris.csv --berlin data/berlin.csv --experiment all
```
Run one part only with `--experiment baselines | labels | features | paris | transfer | figures`
(each needs only the CSVs it uses; `baselines` takes a few minutes and is the quickest demo).
Outputs: tables as CSV and all figures as PNG in `results/` (committed, so results can be inspected
without re-running). Everything is seeded (`SEED = 42`).

| Experiment | Reproduces (paper) | Question answered |
|---|---|---|
| baselines | Table 1 | which model works best on continuous features |
| labels | Table 2, Fig. 1 | raw vs sqrt vs log price, interaction terms |
| features | Tables 3–4 | value of date / categorical / text features |
| paris | Table 5 | does it hold on a second city |
| transfer | Tables 6–7, Fig. 6 | NYC vs Paris vs combined training, tested on Berlin (3 preprocessing variants) |
| figures | Figs. 2–3 | feature histograms and correlation heatmap |

## 4. Repository layout
```
airbnb_price.py   data cleaning, FeatureBuilder, models, all experiments
check_data.py     reports which paper features a CSV contains and how complete they are
make_writeup.py   builds the 2-page PDF write-up (needs reportlab)
results/          CSV tables + figures produced by the experiments
docs/             write-up (PDF) and presentation slides
requirements.txt  Python dependencies
```

## 5. Method in brief
`load_city` (price parsing, cleaning, optional $500 cap) → `split_city` (7:2:1) → `FeatureBuilder`
(continuous, interactions, categorical + multi-hot lists, TF-IDF+SVD text, dates; **fit on train only**) →
`make_model` → `run_experiment` (R², MSE, Spearman, bias, 100-label calibrated R²).
Transfer variants: `paper` (training-city standardisation), `robust` (log1p + clipping), `robust_relgeo`
(+ city-relative coordinates).

## 6. Differences from the paper and the authors' notebooks
* Newer Inside Airbnb snapshot: fee columns and several text columns are absent, so absolute scores differ.
* Encoders, TF-IDF and SVD are fit on the training split only (the authors' notebooks fit on all rows).
* Added an Adam network mirroring the authors' PyTorch model, robust/relative-coordinate preprocessing,
  and Spearman / bias / calibration diagnostics.
