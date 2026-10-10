"""
Predicting Airbnb Listing Price Across Cities (NYC, Paris, Berlin)

Pipeline overview:
1. load_city()        -> parse "$1,200.00" strings, drop bad prices, optional $500 cap
2. split_city()       -> 7:2:1 train/val/test split
3. FeatureBuilder     -> builds the 5 feature groups (fit on TRAIN only, so no leakage):
    C = continuous  (impute, drop correlated, standardise)
    I = 2-degree interaction terms of the continuous features
    O = categorical (one-hot) + list features (amenities, host_verifications)
    T = text        (TF-IDF on uni+bigrams -> TruncatedSVD to 50 dims per column)
    D = date        (days since earliest date, mean-imputed)
4. make_model()       -> Ridge, KNN, RandomForest, XGBoost, MLP (neural network)
5. run_experiment()   -> fit, evaluate MSE and R^2 on train/val/test
6. exp_*()            -> one function per results table

Usage:
python airbnb_price.py --nyc data/nyc.csv --paris data/paris.csv --berlin data/berlin.csv --experiment all

"""

import argparse
import os
import re
import warnings
from collections import Counter

import matplotlib
matplotlib.use("Agg")                      # we only save figures
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import TransformedTargetRegressor
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:                        # keeps the script runnable without xgboost
    HAS_XGB = False

SEED = 42
PRICE_CAP = 500                           


# Column definitions (Kaggle 96-column schema)

MONEY_COLS = ["price", "security_deposit", "cleaning_fee", "extra_people"]
ZERO_FILL = ["security_deposit", "cleaning_fee"]   # missing fee == no fee -> 0
CONTINUOUS = [
    "accommodates", "bathrooms", "bedrooms", "beds", "guests_included",
    "extra_people", "minimum_nights", "maximum_nights",
    "availability_30", "availability_60", "availability_90", "availability_365",
    "number_of_reviews", "reviews_per_month", "review_scores_rating",
    "review_scores_accuracy", "review_scores_cleanliness", "review_scores_checkin",
    "review_scores_communication", "review_scores_location", "review_scores_value",
    "latitude", "longitude", "host_listings_count",
    "calculated_host_listings_count", "security_deposit", "cleaning_fee",
]
CATEGORICAL = [
    "room_type", "property_type", "bed_type", "cancellation_policy",
    "host_is_superhost", "host_identity_verified", "instant_bookable",
    "is_location_exact", "require_guest_profile_picture",
    "require_guest_phone_verification", "host_response_time",
    "neighbourhood_cleansed",
]
LIST_COLS = ["amenities", "host_verifications"]
TEXT_COLS = [
    "name", "summary", "space", "description", "neighborhood_overview", "notes",
    "transit", "access", "interaction", "house_rules", "host_about",
]
DATE_COLS = ["host_since", "first_review", "last_review"]
# Heavy-tailed count columns (e.g. maximum_nights can be 2,147,483,647). Used by robust mode.
SKEWED = ["minimum_nights", "maximum_nights", "number_of_reviews", "reviews_per_month",
          "host_listings_count", "calculated_host_listings_count"]
# Feature-pipeline variants compared in the transfer experiment (see exp_transfer)
VARIANTS = {
    "paper": dict(robust=False, geo="raw"),             # faithful to the paper
    "robust": dict(robust=True, geo="raw"),             # + log1p on skewed counts, clip z-scores
    "robust_relgeo": dict(robust=True, geo="relative"), # + city-relative coordinates (km)
}

# Label transformations: (forward applied to price, inverse). log was found as best 
TRANSFORMS = {
    "raw": (lambda p: p, lambda y: y),
    "sqrt": (np.sqrt, np.square),
    "log": (np.log, np.exp),
}


# 1-2. Loading and splitting

def load_city(path, cap=PRICE_CAP, city=None):
    """Read a listings CSV, turn money strings into floats, and clean the label."""
    df = pd.read_csv(path, low_memory=False)
    # Tag every row with its city (used for city-relative coordinates); overwrites any old column
    df["city"] = city or os.path.splitext(os.path.basename(path))[0]
    for c in MONEY_COLS:
        if c in df.columns:
            # "$1,200.00" -> 1200.0 ; anything unparsable becomes NaN
            cleaned = df[c].astype(str).str.replace(r"[$,]", "", regex=True)
            df[c] = pd.to_numeric(cleaned, errors="coerce")
    # Newer Inside Airbnb snapshots replaced numeric `bathrooms` with text such as "1.5 baths" / "Half-bath" in `bathrooms_text`; recover the number from it.
    if "bathrooms_text" in df.columns:
        txt = df["bathrooms_text"].astype(str)
        num = txt.str.extract(r"(\d+\.?\d*)")[0].astype(float)
        num[txt.str.contains("half", case=False) & num.isna()] = 0.5
        df["bathrooms"] = df["bathrooms"].fillna(num) if "bathrooms" in df.columns else num
    df = df[df["price"].notna() & (df["price"] > 0)]    # log(0) is undefined -> drop
    if cap is not None:
        df = df[df["price"] <= cap]                      # "data thresholding"
    return df.reset_index(drop=True)


def split_city(df):
    """7:2:1 train/val/test split (seeded for reproducibility)."""
    train, rest = train_test_split(df, test_size=0.3, random_state=SEED)
    val, test = train_test_split(rest, test_size=1 / 3, random_state=SEED)  # 0.2 / 0.1 overall
    return train, val, test



# 3. Feature engineering

def parse_list(cell):
    """'{TV,"Cable TV",Wifi}' or "['email','phone']" -> ['TV','Cable TV','Wifi']."""
    if pd.isna(cell):
        return []
    stripped = re.sub(r"[\{\}\[\]'\"]", "", str(cell))
    return [t.strip() for t in stripped.split(",") if t.strip()]


class FeatureBuilder:
    """
    Fit on training data, then transform any split. Fitting only on train means
    means/vocabularies/SVD never see validation or test rows (no data leakage).
    Every column access uses reindex(), so a city missing a column (e.g. Berlin
    lacking a neighbourhood field) just gets NaN instead of crashing -- this is
    what makes cross-city transfer possible.
    """

    def __init__(self, groups, corr_thresh=0.9, svd_dim=50, list_vocab=50,
                 robust=False, geo="raw", clip=5.0):
        """
        robust : log1p the heavy-tailed count columns and clip standardised values to
                 +-`clip` sigma. Stops one absurd value (or a city never seen in training)
                 from producing z-scores in the thousands that blow up SGD training.
        geo    : "raw"      -> latitude/longitude as given (paper)
                 "relative" -> km north/east of the city's median listing + distance to it,
                               so coordinates mean the same thing in every city.
        """
        self.robust, self.geo, self.clip = robust, geo, clip
        self.groups = set(groups)          # subset of {"C","I","O","T","D"}
        self.corr_thresh = corr_thresh
        self.svd_dim = svd_dim
        self.list_vocab = list_vocab

    # helpers 
    def _cont_raw(self, df):
        X = df.reindex(columns=CONTINUOUS).apply(pd.to_numeric, errors="coerce")
        if self.geo == "relative":
            city = df["city"] if "city" in df.columns else pd.Series("all", index=df.index)
            lat, lon = X["latitude"], X["longitude"]
            mid_lat = lat.groupby(city).transform("median")      # per-city "centre"
            mid_lon = lon.groupby(city).transform("median")
            dy = (lat - mid_lat) * 111.0                         # 1 degree latitude ~ 111 km
            dx = (lon - mid_lon) * 111.0 * np.cos(np.radians(mid_lat))
            X["latitude"], X["longitude"] = dy, dx               # now "km north / km east"
            X["dist_center_km"] = np.hypot(dx, dy)
        if self.robust:
            for c in SKEWED:
                X[c] = np.log1p(X[c].clip(lower=0))              # clips 2-billion-night outliers
        return X

    def _scale(self, X):
        Xs = self.scaler.transform(X[self.cont_cols])
        return np.clip(Xs, -self.clip, self.clip) if self.robust else Xs

    def _dates_to_days(self, df):
        raw = df.reindex(columns=DATE_COLS).apply(pd.to_datetime, errors="coerce")
        return pd.DataFrame({c: (raw[c] - self.date_origin).dt.days for c in DATE_COLS})

    # fit
    def fit(self, df):
        g = self.groups

        if g & {"C", "I"}:
            X = self._cont_raw(df).dropna(axis=1, how="all")  # skip columns this city lacks entirely
            self.cont_means = X.mean()
            for c in ZERO_FILL:                              # fees: NaN -> 0, others: mean
                if c in self.cont_means.index:
                    self.cont_means[c] = 0.0
            X = X.fillna(self.cont_means)
            # Drop one column from every highly-correlated pair (in the paper, there were 28 features kept)
            corr = X.corr().abs()
            upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
            self.cont_cols = [c for c in X.columns if not (upper[c] > self.corr_thresh).any()]
            self.scaler = StandardScaler().fit(X[self.cont_cols])
            if "I" in g:
                self.poly = PolynomialFeatures(2, interaction_only=True, include_bias=False)
                self.poly.fit(self._scale(X))

        if "O" in g:
            self.cat_cols = [c for c in CATEGORICAL if c in df.columns]
            try:
                self.ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            except TypeError:                                # sklearn < 1.2
                self.ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
            self.ohe.fit(df[self.cat_cols].astype(str))      # NaN -> "nan" category
            # Multi-hot vocab for list columns: keep the most frequent items on TRAIN
            self.vocab = {}
            for col in LIST_COLS:
                if col in df.columns:
                    counts = Counter(t for cell in df[col] for t in parse_list(cell))
                    self.vocab[col] = [t for t, _ in counts.most_common(self.list_vocab)]

        if "T" in g:
            self.text_models = {}
            for col in TEXT_COLS:
                if col not in df.columns:
                    continue
                texts = df[col].fillna("").astype(str)
                if texts.str.len().sum() == 0:
                    continue
                tfidf = TfidfVectorizer(ngram_range=(1, 2), stop_words="english",
                                        min_df=3, max_features=20000)
                M = tfidf.fit_transform(texts)
                k = min(self.svd_dim, M.shape[1] - 1)        # SVD needs k < n_features
                if k < 1:
                    continue
                svd = TruncatedSVD(n_components=k, random_state=SEED).fit(M)
                self.text_models[col] = (tfidf, svd)

        if "D" in g:
            raw = df.reindex(columns=DATE_COLS).apply(pd.to_datetime, errors="coerce")
            self.date_origin = raw.min().min()               # earliest date across all 3 cols
            days = self._dates_to_days(df)
            self.date_means = days.mean().fillna(0)
            self.date_scaler = StandardScaler().fit(days.fillna(self.date_means))
        return self

    # transform
    def transform(self, df):
        g, blocks = self.groups, []

        if g & {"C", "I"}:
            X = self._cont_raw(df).fillna(self.cont_means)[self.cont_cols]
            Xs = self._scale(X)
            blocks.append(self.poly.transform(Xs) if "I" in g else Xs)

        if "O" in g:
            blocks.append(self.ohe.transform(df.reindex(columns=self.cat_cols).astype(str)))
            for col, vocab in self.vocab.items():
                index = {t: i for i, t in enumerate(vocab)}
                M = np.zeros((len(df), len(vocab)), dtype=np.float32)
                for r, cell in enumerate(df[col] if col in df else [None] * len(df)):
                    for t in parse_list(cell):
                        if t in index:
                            M[r, index[t]] = 1.0
                blocks.append(M)

        if "T" in g:
            for col, (tfidf, svd) in self.text_models.items():
                texts = df[col].fillna("").astype(str) if col in df else pd.Series([""] * len(df))
                blocks.append(svd.transform(tfidf.transform(texts)))

        if "D" in g:
            days = self._dates_to_days(df).fillna(self.date_means)
            blocks.append(self.date_scaler.transform(days))

        if not blocks:
            raise ValueError("No feature groups selected.")
        return np.hstack(blocks).astype(np.float32)


# 4. Models

def make_model(name):
    if name == "ridge":                    # linear regression with L2 penalty
        return Ridge(alpha=1.0)
    if name == "knn":                      # unweighted KNN, Euclidean distance
        return KNeighborsRegressor(n_neighbors=5)
    if name == "rf":
        return RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=SEED)
    if name == "xgb":
        if HAS_XGB:
            return XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1,
                                tree_method="hist", n_jobs=-1, random_state=SEED)
        warnings.warn("xgboost not installed: using sklearn HistGradientBoosting instead.")
        return HistGradientBoostingRegressor(max_iter=300, random_state=SEED)
    if name == "nn":
        mlp = MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu", solver="sgd",
                           nesterovs_momentum=True, momentum=0.9, learning_rate_init=0.005,
                           alpha=0.005, batch_size=64, early_stopping=True,
                           validation_fraction=0.1, n_iter_no_change=10, tol=1e-4,
                           max_iter=200, random_state=SEED)
        return TransformedTargetRegressor(regressor=mlp, transformer=StandardScaler())
    if name == "nn_adam":
        mlp = MLPRegressor(hidden_layer_sizes=(128, 512, 64), activation="relu", solver="adam",
                           learning_rate_init=1e-4, alpha=1e-5, batch_size=64, max_iter=100,
                           early_stopping=True, validation_fraction=0.1, n_iter_no_change=10,
                           random_state=SEED)
        return TransformedTargetRegressor(regressor=mlp, transformer=StandardScaler())
    raise ValueError(f"unknown model {name}")


# 5. Experiment runner

def scatter(y_true, y_pred, path, title):
    """Predicted-vs-truth plot (paper Figures 1 and 6)."""
    plt.figure(figsize=(4.5, 4.5))
    plt.scatter(y_true, y_pred, s=3, alpha=0.4)
    lim = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lim, lim, "r--", lw=1)
    plt.xlabel("Ground-truth label"); plt.ylabel("Predicted label"); plt.title(title, fontsize=9)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


PREDS = {}   # (tag, model, groups, label, split) -> (y_true, y_pred); lets us draw panels


def panel_figure(items, path):
    """Predicted-vs-true panels side by side, e.g. paper Fig. 1 (a)/(b) and Fig. 6 (a)/(b).
    items = [(PREDS key, panel title), ...]. Axes are limited to the true-label range so a
    diverged model's wild predictions don't squash the plot (those points are simply off-scale)."""
    fig, axes = plt.subplots(1, len(items), figsize=(4.6 * len(items), 4.4))
    for ax, (key, title) in zip(np.atleast_1d(axes), items):
        y, p = PREDS[key]
        pad = 0.05 * (y.max() - y.min())
        lim = [y.min() - pad, y.max() + pad]
        ax.scatter(y, p, s=3, alpha=0.4)
        ax.plot(lim, lim, "r--", lw=1)                       # perfect-prediction line
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("Ground-truth label"); ax.set_ylabel("Predicted label")
        ax.set_title(title, fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def run_experiment(tag, model_name, groups, label, train_sets, eval_sets, out_dir=None,
                   fb_kwargs=None, extra=None):
    """
    Train on `train_sets` (list of DataFrames, concatenated), evaluate on train + every
    entry of `eval_sets`. MSE and R^2 are computed in the *transformed* label space (e.g. log-price MSE ~0.13).
    """
    train_df = pd.concat(train_sets, ignore_index=True)
    fwd, _ = TRANSFORMS[label]
    fb = FeatureBuilder(groups.split(",") if isinstance(groups, str) else groups,
                        **(fb_kwargs or {})).fit(train_df)
    X_train, y_train = fb.transform(train_df), fwd(train_df["price"].values)

    model = make_model(model_name).fit(X_train, y_train)

    row = {"experiment": tag, "model": model_name, "features": groups, "label": label,
           "n_features": X_train.shape[1], **(extra or {})}
    for split, df in {"train": train_df, **eval_sets}.items():
        X = X_train if split == "train" else fb.transform(df)
        y = fwd(df["price"].values)
        pred = model.predict(X)
        row[f"{split}_mse"] = mean_squared_error(y, pred)
        row[f"{split}_r2"] = r2_score(y, pred)
        # --- diagnostics that separate "ranks listings well" from "gets the price LEVEL right" ---
        row[f"{split}_bias"] = float(np.mean(pred - y))              # >0: over-predicts on average
        row[f"{split}_spearman"] = float(spearmanr(y, pred)[0])      # rank agreement, ignores level
        # Few-shot calibration: estimate the level shift from just 100 labelled listings of this
        # split, add it to all predictions, and score on the remaining listings.
        idx = np.random.default_rng(SEED).permutation(len(y))
        k = min(100, len(y) // 2)
        offset = float(np.mean(y[idx[:k]] - pred[idx[:k]]))
        row[f"{split}_calib100_r2"] = r2_score(y[idx[k:]], pred[idx[k:]] + offset)
        if out_dir and split != "train":
            # filename includes features+label so configs of one table don't overwrite each other
            fname = f"{tag}_{model_name}_{groups}_{label}_{split}.png".replace(",", "+")
            scatter(y, pred, os.path.join(out_dir, fname), f"{tag} | {model_name} | {split}")
            PREDS[(tag, model_name, str(groups), label, split)] = (y, pred)
    print(f"[{tag}] {model_name:5s} {groups:8s} {label:4s} "
          f"train R2={row['train_r2']:.3f}  " +
          "  ".join(f"{k[:-3]} R2={v:.3f}" for k, v in row.items()
                    if k.endswith("_r2") and k != "train_r2" and "calib" not in k))
    return row


# 6. Experiments (mapped to the tables)
def exp_baselines(nyc, out):
    """Table 1: five models on continuous features, price capped at $500, raw label."""
    tr, va, te = split_city(load_city(nyc))
    rows = [run_experiment("table1", m, "C", "raw", [tr], {"val": va, "test": te}, out)
            for m in ["ridge", "knn", "rf", "nn", "nn_adam", "xgb"]]
    # Bar chart of test R^2 per model (handy for the slides)
    plt.figure(figsize=(6, 3.8))
    plt.bar([r["model"] for r in rows], [r["test_r2"] for r in rows], color="tab:blue")
    plt.ylabel("Test R$^2$"); plt.title("NYC, continuous features, price <= $500")
    plt.tight_layout(); plt.savefig(os.path.join(out, "figure_model_comparison.png"), dpi=150)
    plt.close()
    return rows


def exp_labels(nyc, out):
    """Table 2: label transformations with XGBoost. NO price cap -> shows why log helps."""
    tr, va, te = split_city(load_city(nyc, cap=None))
    configs = [("C", "raw"), ("C", "sqrt"), ("C", "log"), ("C,I", "log")]
    rows = [run_experiment("table2", "xgb", g, lab, [tr], {"val": va, "test": te}, out)
            for g, lab in configs]
    # Paper Figure 1: (a) no label transformation vs (b) log transformation
    panel_figure([(("table2", "xgb", "C", "raw", "test"), "(a) No transformation"),
                  (("table2", "xgb", "C", "log", "test"), "(b) Log transformation")],
                 os.path.join(out, "figure1_label_transformation.png"))
    return rows


def exp_features(nyc, out):
    """Tables 3-4: contribution of each feature group (XGBoost, capped)."""
    tr, va, te = split_city(load_city(nyc))
    configs = [("D", "raw"), ("O", "raw"), ("T", "raw"), ("C,O,T", "raw"), ("C,O,T", "log")]
    return [run_experiment("table4", "xgb", g, lab, [tr], {"val": va, "test": te}, out)
            for g, lab in configs]


def exp_paris(paris, out):
    """Table 5: same XGBoost setting on the Paris dataset."""
    tr, va, te = split_city(load_city(paris))
    return [run_experiment("table5", "xgb", "C,O,T", lab, [tr], {"val": va, "test": te}, out)
            for lab in ["raw", "log"]]


def exp_figures(nyc, out):
    """Paper Figure 2 (histograms of continuous features) and Figure 3 (correlation heatmap)."""
    df = load_city(nyc, city="nyc")
    show = [c for c in ["accommodates", "bedrooms", "latitude", "longitude",
                        "number_of_reviews", "review_scores_value"] if c in df.columns]
    ax = df[show].apply(pd.to_numeric, errors="coerce").hist(bins=20, figsize=(8, 6))
    plt.tight_layout(); plt.savefig(os.path.join(out, "figure2_histograms.png"), dpi=150)
    plt.close("all")

    X = df.reindex(columns=CONTINUOUS).apply(pd.to_numeric, errors="coerce")
    X = X.dropna(axis=1, how="all")                      # skip columns absent from this snapshot
    corr = X.corr()
    plt.figure(figsize=(8, 7))
    plt.imshow(corr, cmap="viridis", vmin=-1, vmax=1)    # yellow = high, blue = low (as in paper)
    plt.xticks(range(len(corr)), corr.columns, rotation=90, fontsize=6)
    plt.yticks(range(len(corr)), corr.columns, fontsize=6)
    plt.colorbar(label="correlation"); plt.title("Continuous feature correlation (NYC)")
    plt.tight_layout(); plt.savefig(os.path.join(out, "figure3_correlation.png"), dpi=150)
    plt.close("all")
    return [{"figure": "figure2_histograms.png"}, {"figure": "figure3_correlation.png"}]


def exp_transfer(nyc, paris, berlin, out):
    """
    Tables 6-7: neural net trained on NYC / Paris / NYC+Paris (C+O features, log label),
    tested on (a) its own cities' test split and (b) the unseen Berlin dataset.
    Every setup is run under three feature-pipeline VARIANTS so the write-up can show WHY
    naive transfer fails (see VARIANTS at the top of the file).
    """
    ny = split_city(load_city(nyc, city="nyc"))
    pa = split_city(load_city(paris, city="paris"))
    berlin_all = load_city(berlin, city="berlin")
    setups = {"nyc": ([ny[0]], [ny[2]]),
              "paris": ([pa[0]], [pa[2]]),
              "nyc_paris": ([ny[0], pa[0]], [ny[2], pa[2]])}
    rows = []
    for variant, kwargs in VARIANTS.items():
        for name, (train_sets, test_sets) in setups.items():
            test_df = pd.concat(test_sets, ignore_index=True)
            rows.append(run_experiment(f"table6_7_{variant}_{name}", "nn", "C,O", "log",
                                       train_sets, {"test": test_df, "berlin": berlin_all},
                                       out, fb_kwargs=kwargs,
                                       extra={"variant": variant, "train_on": name}))
    # Paper Figure 6 for every variant: (a) train NYC only vs (b) train NYC+Paris, test Berlin
    for variant in VARIANTS:
        panel_figure([((f"table6_7_{variant}_nyc", "nn", "C,O", "log", "berlin"), "(a) Train NYC, test Berlin"),
                      ((f"table6_7_{variant}_nyc_paris", "nn", "C,O", "log", "berlin"),
                       "(b) Train NYC+Paris, test Berlin")],
                     os.path.join(out, f"figure6_transfer_{variant}.png"))
    # Summary: ranking (Spearman) vs raw R^2 vs R^2 after 100-label calibration, on Berlin
    d = pd.DataFrame(rows)
    x, w = np.arange(len(d)), 0.27
    plt.figure(figsize=(11, 4.6))
    plt.bar(x - w, d["berlin_r2"].clip(lower=-1), w, label="R$^2$ (clipped at -1)")
    plt.bar(x, d["berlin_spearman"], w, label="Spearman (ranking)")
    plt.bar(x + w, d["berlin_calib100_r2"], w, label="R$^2$ after 100-label calibration")
    plt.xticks(x, d["variant"] + "\n" + d["train_on"], fontsize=7)
    plt.axhline(0, color="k", lw=0.5); plt.legend(fontsize=8); plt.title("Transfer to Berlin")
    plt.tight_layout(); plt.savefig(os.path.join(out, "figure_transfer_summary.png"), dpi=150)
    plt.close()
    return rows


# MAIN 
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--nyc"); ap.add_argument("--paris"); ap.add_argument("--berlin")
    ap.add_argument("--experiment", default="all",
                    choices=["baselines", "labels", "features", "paris", "transfer", "figures", "all"])
    ap.add_argument("--out", default="results", help="folder for CSVs and scatter plots")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    plan = {
        "baselines": (lambda: exp_baselines(a.nyc, a.out), ["nyc"]),
        "labels": (lambda: exp_labels(a.nyc, a.out), ["nyc"]),
        "features": (lambda: exp_features(a.nyc, a.out), ["nyc"]),
        "paris": (lambda: exp_paris(a.paris, a.out), ["paris"]),
        "figures": (lambda: exp_figures(a.nyc, a.out), ["nyc"]),
        "transfer": (lambda: exp_transfer(a.nyc, a.paris, a.berlin, a.out), ["nyc", "paris", "berlin"]),
    }
    chosen = list(plan) if a.experiment == "all" else [a.experiment]
    for name in chosen:
        run, needs = plan[name]
        if any(getattr(a, n) is None for n in needs):
            print(f"-- skipping '{name}': needs --{', --'.join(needs)}")
            continue
        df = pd.DataFrame(run())
        df.to_csv(os.path.join(a.out, f"{name}.csv"), index=False)
        print(f"\n== {name} ==\n{df.round(4).to_string(index=False)}\n")


if __name__ == "__main__":
    main()
