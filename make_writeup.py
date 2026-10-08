"""
Builds the 2-page project write-up (PDF) with reportlab.

    pip install reportlab
    python make_writeup.py            # writes writeup.pdf next to this script

1. Edit the CONFIG block below (names, SRNs, repo URL, snapshot date).
2. If results/figure_transfer_summary.png exists (created by airbnb_price.py), it is embedded.
3. All numbers come from the team's own full run (--experiment all); if you re-run with a
   different seed or snapshot, update the tables in the RESULTS section.
"""
import os
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

# ------------------------------- CONFIG: edit these --------------------------------------
TEAM = "Vismay Chandra Dev (PES2UG24CS597), Vishwas Hosamath (PES2UG24CS596)"
SECTION = "5J"
REPO = "https://github.com/<user>/<repo>"
SNAPSHOT = "07-10-2026"
FIG = "results/figure_transfer_summary.png"
OUT = "writeup.pdf"
# -----------------------------------------------------------------------------------------

ss = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=ss["Normal"], fontName="Helvetica", fontSize=8.2,
                      leading=10.1, alignment=TA_JUSTIFY, spaceAfter=2.5)
H = ParagraphStyle("h", parent=BODY, fontName="Helvetica-Bold", fontSize=9.4, leading=11.5,
                   spaceBefore=4.5, spaceAfter=2, textColor=colors.HexColor("#1F3A5F"))
TITLE = ParagraphStyle("t", parent=BODY, fontName="Helvetica-Bold", fontSize=13.5, leading=16,
                       alignment=0, spaceAfter=1)
SUB = ParagraphStyle("s", parent=BODY, fontSize=8, leading=10, alignment=0, textColor=colors.HexColor("#444444"))
BUL = ParagraphStyle("b", parent=BODY, leftIndent=9, bulletIndent=0, spaceAfter=1.5)
CAP = ParagraphStyle("c", parent=BODY, fontSize=7.3, leading=8.8, alignment=0, textColor=colors.HexColor("#333333"))
R2 = "R<super>2</super>"


def table(rows, widths, bold_rows=()):
    rows = [[str(c).replace("R2", "R\u00b2") for c in row] for row in rows]
    t = Table(rows, colWidths=[w * cm for w in widths], hAlign="LEFT")
    style = [("FONT", (0, 0), (-1, -1), "Helvetica", 7.3),
             ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 7.3),
             ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E6ECF4")),
             ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#9AA5B1")),
             ("TOPPADDING", (0, 0), (-1, -1), 1.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.2),
             ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
             ("ALIGN", (1, 1), (-1, -1), "CENTER")]
    for r in bold_rows:
        style.append(("FONT", (0, r), (-1, r), "Helvetica-Bold", 7.3))
    t.setStyle(TableStyle(style))
    return t


def P(text, style=BODY):
    return Paragraph(text.replace("R2", "R<super>2</super>"), style)


story = [
    P("Predicting Airbnb Listing Price Across Cities (NYC, Paris, Berlin)", TITLE),
    P(f"UE24CS352A Machine Learning - Mini-Project | {TEAM} | {SECTION} | Code: {escape(REPO)}", SUB),

    P("1. Problem statement", H),
    P("Nightly Airbnb prices are set by hosts by feel, which is hard for new hosts and opaque for guests. "
      "We predict the listing price from listing attributes in three cities and ask two questions: "
      "(i) which regression methods and feature groups work best, and (ii) does a model trained on some "
      "cities transfer to an unseen one? The task follows Luo, Zhou and Zhou (2019); we re-implemented "
      "it independently and consulted the authors' notebooks for reference."),

    P("2. Dataset", H),
    P(f"We use the detailed <i>Inside Airbnb</i> listing tables for New York City, Paris and Berlin (snapshot {SNAPSHOT}; "
      "90 columns). Raw rows: NYC 30,259, Paris 77,679, Berlin 12,776. After removing listings with a missing or zero "
      "price and those above $500/night, <b>19,148 (NYC), 42,089 (Paris) and 8,259 (Berlin)</b> remain "
      "(the label-transformation experiment keeps the uncapped data). Each city is split 7:2:1 into train/validation/test "
      "(seed 42); Berlin is used only as an unseen test city. Our snapshot is newer than the paper's and lacks "
      "<i>security_deposit, cleaning_fee, guests_included, extra_people, summary, transit, house_rules</i> and several other text "
      "columns; <i>instant_bookable, host_response_time, host_verifications, host_since</i> and <i>neighborhood_overview</i> "
      "are present but empty. Absolute scores are therefore not directly comparable with the paper."),

    P("3. Approach", H),
    P("<b>Features</b> (all statistics fit on the training split only): <b>C</b> continuous - 21 columns after dropping "
      "absent and highly correlated ones (|r| &gt; 0.9), mean-imputed, standardised; <b>I</b> 2-degree interaction terms; "
      "<b>O</b> one-hot categoricals plus multi-hot amenities (top 50), 339 columns for NYC; <b>T</b> TF-IDF (uni+bigrams) of "
      "<i>name, description, host_about</i>, each reduced to 50 dimensions by truncated SVD; <b>D</b> date features "
      "(days since the earliest date). <b>Label:</b> raw price, square root or log; metrics are computed in the transformed space. "
      "<b>Models:</b> Ridge, KNN (k=5), Random Forest (100 trees), XGBoost (300 trees, depth 6), and two neural networks: "
      "an SGD+Nesterov MLP (64-32, lr 0.005, L2 0.005, early stopping) and an Adam MLP (128-512-64, lr 1e-4, weight decay 1e-5). "
      f"<b>Metrics:</b> {R2}, MSE, plus Spearman correlation, mean bias, and {R2} after calibrating the price level on 100 labelled "
      "target-city listings (to separate ranking quality from price-level shift)."),
    P("<b>Transfer experiments.</b> A neural network (C+O features, log label) is trained on NYC, Paris, or NYC+Paris and tested on "
      "Berlin under three preprocessing variants: <b>paper</b> (standardise with training statistics), <b>robust</b> "
      "(log1p on heavy-tailed counts and clip standardised values to +-5 sigma) and <b>robust+relgeo</b> (additionally replace "
      "latitude/longitude by km north/east of the city's median listing and distance to it)."),

    P("4. Implementation overview", H),
    P("A single Python module (<i>airbnb_price.py</i>) implements <i>load_city</i> (price parsing, cleaning, cap) &rarr; "
      "<i>split_city</i> &rarr; <i>FeatureBuilder</i> (fit/transform; missing columns become NaN so cities with different schemas "
      "can share one pipeline) &rarr; <i>make_model</i> &rarr; <i>run_experiment</i>, with one function per paper table; "
      "<i>check_data.py</i> audits which paper features a CSV provides. Outputs (CSV tables and figures) go to <i>results/</i>. "
      "Compared with the authors' notebooks we fit TF-IDF/SVD/encoders on the training split only (they fit on all rows), added "
      "the Adam network to mirror their PyTorch model, and added the robust/relative-coordinate variants and diagnostics."),

    P("5. Results", H),
    P("<b>Table 1 - NYC, continuous features, price &lt;= $500, raw price.</b>", CAP),
    table([["Model", "Ridge", "KNN", "Random Forest", "NN (SGD 64-32)", "NN (Adam 128-512-64)", "XGBoost"],
           ["Train " + "R2", "0.423", "0.731", "0.960", "0.723", "0.743", "0.892"],
           ["Test R2", "0.446", "0.607", "0.717", "0.666", "0.686", "0.740"],
           ["Test MSE", "7199", "5109", "3685", "4344", "4087", "3379"]],
          [2.2, 1.9, 1.9, 2.6, 3.0, 3.6, 2.0], bold_rows=(2,)),
    Spacer(1, 3),
    P("<b>Table 2 - XGBoost test R2 by label transform and feature group.</b>", CAP),
    table([["NYC, uncapped, C features", "raw", "sqrt", "log", "log + interactions"],
           ["Test R2", "0.470", "0.689", "0.808", "0.803"]],
          [4.3, 2.2, 2.2, 2.2, 3.6]),
    Spacer(1, 2),
    table([["NYC, capped", "D only", "O only", "T only", "C+O+T (raw)", "C+O+T (log)", "Paris C+O+T (raw / log)"],
           ["Test R2", "0.005", "0.498", "0.442", "0.766", "0.825", "0.734 / 0.826"]],
          [2.6, 1.7, 1.7, 1.7, 2.5, 2.5, 4.3], bold_rows=()),
    Spacer(1, 3),
    P("<b>Table 3 - Neural-network transfer to Berlin</b> (in-city = test split of the training cities; calibrated = 100 Berlin labels).", CAP),
    table([["Train on", "Variant", "In-city R2", "Berlin R2", "Berlin Spearman", "Berlin R2 calibrated"],
           ["NYC", "paper", "0.792", "-7922.7", "0.801", "0.613"],
           ["Paris", "paper", "0.763", "-2317.5", "0.559", "0.291"],
           ["NYC+Paris", "paper", "0.781", "0.812", "0.872", "0.827"],
           ["NYC", "robust", "0.793", "0.604", "0.790", "0.716"],
           ["Paris", "robust", "0.789", "-1.814", "0.838", "0.710"],
           ["NYC+Paris", "robust", "0.785", "0.822", "0.877", "0.843"],
           ["NYC", "robust+relgeo", "0.802", "0.058", "0.854", "0.801"],
           ["Paris", "robust+relgeo", "0.794", "0.752", "0.839", "0.771"],
           ["NYC+Paris", "robust+relgeo", "0.791", "0.481", "0.866", "0.816"]],
          [2.3, 2.6, 2.1, 2.1, 3.0, 3.6]),
]

if os.path.exists(FIG):
    story += [Spacer(1, 3), Image(FIG, width=11.5 * cm, height=4.8 * cm, hAlign="LEFT"),
              P("Figure 1 - Berlin transfer: raw R2 (clipped at -1), Spearman, and R2 after 100-label calibration.", CAP)]

story += [
    P("6. Conclusions", H),
    P("&bull; <b>Model ranking matches the paper:</b> XGBoost (test R2 0.740) &gt; Random Forest (0.717) &gt; neural networks "
      "(0.666 / 0.686) &gt; KNN (0.607) &gt; Ridge (0.446). Forests overfit most (train 0.960 vs test 0.717).", BUL),
    P("&bull; <b>The log label is the biggest single gain</b> (0.470 to 0.808 without a price cap); interaction terms gave no gain "
      "(0.803) and overfit. Categorical and text features help (all groups, log label: 0.825); date features carry no signal (0.005).", BUL),
    P("&bull; <b>Naive transfer fails numerically, not because the ranking is lost.</b> With training-city standardisation, "
      "Berlin coordinates and extreme <i>maximum_nights</i> values reach thousands of standard deviations and the network diverges "
      "(R2 = -7923). Robust preprocessing removes the blow-up.", BUL),
    P("&bull; <b>Ranking transfers, price level does not.</b> Berlin Spearman is 0.79-0.88 for the non-diverged variants, close to "
      "in-city values, while the bias reflects city price levels (NYC-trained relative-coordinate model over-predicts Berlin by about "
      "0.80 log units, roughly 2.2x). Calibrating with only 100 Berlin listings lifts R2 to 0.71-0.84; for NYC to Berlin with relative "
      "coordinates it reaches 0.801, matching NYC's in-city 0.802. Combined NYC+Paris training with raw coordinates scores highest "
      "uncalibrated (0.822), which we suspect (not proven) is because raw coordinates let the network infer a city-level price level.", BUL),
    P("&bull; <b>Limitations:</b> single random seed per run, a newer snapshot with fewer features than the paper, a $500 price cap, and "
      "one random 100-listing calibration draw. Prices are as provided in the files (currency not verified).", BUL),
]

doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=1.5 * cm, rightMargin=1.5 * cm,
                        topMargin=1.2 * cm, bottomMargin=1.1 * cm,
                        title="Predicting Airbnb Listing Price Across Cities", author=TEAM)
doc.build(story)
print("wrote", OUT)
