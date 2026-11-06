# ST5227 — Hotel booking cancellation: project plan

This document maps our analysis to the course brief and rubric (Data description, Comparative numerical experiments, Feature exploration, Unsupervised learning, Limitations). It incorporates exploratory data analysis (EDA) on `hotel_bookings.csv` from the course repository and cites concrete numbers you can reuse in the report.

**Supervised block comes first in execution and write-up; unsupervised learning (PCA, clustering) is last**, with explicit links back to cancellation.

---

## Dataset and supervised task (rubric: Data description — 5 pts)

### Source and business question

- **Data:** Hotel booking demand (Antonio, Almeida & Nunes, 2019, *Data in Brief*), commonly distributed as “Hotel booking demand” on Kaggle. The repository CSV has **119,390 rows** and **32 columns** (header row plus 119,389 bookings).
- **Supervised target:** Binary **`is_canceled`** (0 = not cancelled, 1 = cancelled). Class balance is roughly **63% / 37%** (not cancelled vs cancelled), so neither extreme imbalance nor a trivial majority baseline.
- **Why predict cancellation:** Hotels and revenue managers care about no-shows and late cancellations for overbooking policy, deposit design, staffing, and forecasting. The task is classification with probabilistic outputs (risk scores) where useful.

### Variable types (high level)

| Kind | Examples |
|------|-----------|
| Categorical / string | `hotel` (2 levels), `meal`, `country`, `market_segment`, `distribution_channel`, `reserved_room_type`, `assigned_room_type`, `deposit_type`, `customer_type`, `reservation_status`, `arrival_date_month` |
| Numeric (integer/count) | `lead_time`, night stays, `adults`, `children`, `babies`, repeat and history counts, `booking_changes`, `days_in_waiting_list`, parking and special requests |
| Numeric (continuous) | `adr` (average daily rate, euros) |
| Date-like | `arrival_date_*` fields; `reservation_status_date` (see leakage) |

**High-cardinality IDs:** `agent` and `company` have many levels; `company` is **mostly missing** (see EDA below), which is informative for modelling (e.g. create `has_company` and group rare agents).

---

## EDA summary (feeds Data description + Feature exploration)

### Missing values

- **`children`:** 4 missing — impute with 0 or median and flag if needed.
- **`country`:** 488 missing — use “Unknown” level or impute from segment if justified.
- **`agent`:** 16,340 missing — treat as its own level (e.g. “no agent”) or group.
- **`company`:** 112,593 missing (~94%) — **do not drop rows**; encode missingness (`has_company`) and optionally top-*k* company IDs if any signal remains after grouping.

### Duplicates

- **Exact duplicate rows:** about **32,000** duplicate records on the full 32-column row. Many are likely multiple rooms for the same group booking recorded identically. **Recommendation:** drop duplicate rows for *train/test evaluation* to avoid putting copies of the same latent booking in both train and test (optimistic bias, especially for k-NN and tree ensembles). Document the count dropped and the business interpretation (group bookings).

### Target leakage — must exclude from supervised features

These fields are **post-booking or outcome-equivalent** and would inflate performance unrealistically:

| Column | Why it leaks |
|--------|----------------|
| **`reservation_status`** | Takes values `Canceled`, `Check-Out`, `No-Show` — this **is** the outcome path; it trivially encodes `is_canceled`. |
| **`reservation_status_date`** | For cancelled bookings it aligns with cancellation timing; for completed stays it is checkout — **not known at booking time** for prediction. |

**Optional / debatable leakage (document choice):**

| Column | Issue |
|--------|--------|
| **`assigned_room_type`** | Typically known at check-in, not at booking. If `reserved_room_type ≠ assigned_room_type`, that often implies the guest **arrived** — conditioning on it can leak “did not cancel.” EDA: mismatched room types show a **very different** cancellation mix than matched pairs. **Plan:** exclude `assigned_room_type` from the main model; optionally report a sensitivity analysis. |
| **`country`** | Some documentation suggests country may be updated at check-in. **Plan:** primary models use grouped `country` or drop; run one robustness check without country. |
| **`booking_changes`**, **`days_in_waiting_list`** | They accumulate over the life of the booking; early cancellations have less time to accumulate changes. **Plan:** treat as “softer” leakage — keep for “full information at decision time” narrative or ablate in sensitivity analysis. |

### Outliers and sanity checks

- **`adr`:** Distribution shows a **negative** minimum (at least one row), a **very large maximum** (5,400), and a bulk of rates in a plausible hotel range. **Plan:** cap or remove extreme `adr` (e.g. winsorize at 99th percentile or drop obvious errors) and remove negative `adr`. Report rule in Data description.
- **Guest counts:** Check rows with `adults + children + babies = 0` and remove or fix (EDA found a small number of zero-guest rows).

### Strong bivariate patterns (for Feature exploration figures)

- **`deposit_type`:** “Non Refund” vs “No Deposit” shows **dramatically** different cancellation rates (Non Refund ≈ **near-total cancellation** in the marginal table — verify and discuss: likely a **business rule** confound: non-refundable prepaid bookings may be labelled cancelled when invoicing differs; treat as **high-signal but interpret with care**).
- **`market_segment`, `customer_type`, `hotel`:** Clear differences in marginal cancellation rates (e.g. online vs corporate, city vs resort).
- **`lead_time`:** Cancellation rate tends to increase with longer lead time (bucketed plots for the report).

Document 3–4 **carefully chosen** figures: one on class balance, one on `lead_time` vs cancellation, one on `deposit_type` / `market_segment`, and one correlation or mosaic for key categoricals.

---

## Data cleaning pipeline (rubric: Data description)

1. Drop **`reservation_status`** and **`reservation_status_date`** from all supervised learning features.
2. Drop or separately model **`assigned_room_type`** (default: drop for primary results).
3. Deduplicate rows (document count removed).
4. Impute missing `children` (e.g. 0); encode missing `agent` and `country`.
5. Fix `adr` (remove negative, cap/winsorize extremes).
6. Remove invalid zero-guest rows.
7. Feature engineering (keep simple and justified): total nights, total guests, month as integer, maybe weekend ratio; group rare `country` and `agent` levels (top-*k* + “Other”).
8. **Temporal note:** `arrival_date_year` is useful for **train/test by time** (generalisation to future seasons), but using year as a categorical feature can be debated; plan one **time-based holdout** (e.g. train on 2015–2016, test on 2017) as a robustness check.

---

## Downsampling (rubric: Data description — “intelligent subset”)

Full cleaned data remains **tens of thousands** of rows — enough for stable inference but heavy for **repeated splits + CV + multiple models**.

**Data-driven justification (to write verbatim in the report):**

1. **Learning curve:** Fix a large holdout set; train several model families on increasing subsample sizes (e.g. 2k, 5k, 10k, 20k, …). Plot test ROC–AUC vs training size. Stop increasing *n* when performance **plateaus** and **rankings** among models stabilize.
2. **Stratified subsampling:** When downsampling, stratify on **`hotel` × `arrival` month/year × `is_canceled`** (or similar) so season and hotel mix match the full data.
3. **Separate tuning data:** Optionally hold out a disjoint subset used **only** for hyperparameter search, then run the repeated train/test experiment on another stratified sample, to avoid optimistic bias from tuning on the same points you test.

Target analysis size: **on the order of 25k–40k rows** after cleaning if runtime permits; otherwise justify **~15k–20k** using the learning curve.

---

## Comparative numerical experiments (rubric: Comparative numerical experiments — 30 pts)

### Model zoo (diverse mechanisms)

| Model | Role |
|-------|------|
| **Baseline** | Majority class or intercept-only logistic (reference metrics). |
| **Logistic regression** | Linear boundary, interpretable coefficients; use **regularised** version (e.g. L1/L2 or elastic net via `l1_ratio` / `C` in scikit-learn 1.8+). |
| **GAM** | Additive smooths (e.g. `pygam` or splines) for 3–5 key continuous variables (`lead_time`, `adr`, nights, …); tests **nonlinearity** while staying interpretable. |
| **Tree ensembles** | **Random forest** (bagging + interactions) and **gradient boosting** (sequential, strong tabular performance). |
| **Optional** | **k-NN** (local geometry) or a **shallow single tree** (max depth constrained) as an interpretable nonlinear contrast. |

This selection addresses rubric item “good selection of different models with different properties.”

### Hyperparameters

- Use **stratified k-fold CV** (e.g. *k* = 5) on the **tuning subset** for parameters that materially affect performance: `C` for logistic, depth / leaves / learning rate for boosting, `min_samples_leaf` / `max_features` for forest, spline smoothing / `lam` for GAM.
- Fixed defaults with short justification are acceptable for minor knobs (e.g. number of trees in RF if monotonic in *n_trees*).

### Robust comparison design

1. **Repeated stratified train/test splits** (e.g. 70/30, **15–25 repeats**, different seeds): report distribution of **ROC–AUC**, **Brier score**, **log loss**, and **accuracy** at 0.5 threshold.
2. **Paired analysis:** Same splits for all models → paired tests on AUC differences.
3. **Corrected resampled *t*-test** (Nadeau & Bengio): accounts for overlapping training data across splits when comparing two models; report alongside a **Wilcoxon signed-rank** on paired AUCs.
4. **Multiple comparisons:** apply **Holm** (or Bonferroni) across planned pairwise contrasts (e.g. boosting vs GAM, forest vs logistic, etc.).
5. **Key substantive questions:** (a) Does a flexible model beat linear? (b) Is the gain from **interactions** (deep trees vs GAM/additive)? (c) If interpretability matters, what is the **performance cost** of GAM vs best ensemble?

### Key conclusions (primary takeaways — rubric 15 pts)

Draft bullets to fill after experiments:

- Best-performing model and **effect size** (mean ΔAUC ± SE across repeats).
- Whether nonlinear/additive structure is **material** (GAM vs logistic; partial dependence).
- Whether ensembles meaningfully beat GAM **after** corrected tests.
- Calibration narrative (Brier / log loss), not only AUC.

---

## Feature exploration (rubric: Feature exploration — 5 pts)

**Methods (pick several, justify):**

- **Permutation importance** (forest or boosting holdout).
- **Model-agnostic:** coefficients from regularised logistic (with grouped one-hot interpretation).
- **GAM partial dependence** / shape plots for `lead_time`, `adr`, stays.
- Optional: **SHAP** for boosting (if time); partial dependence **two-way** for `lead_time × deposit_type`.

**Selection rule:** Start from domain (pricing, lead time, channel), then confirm with importance. Report **uncertainty** where feasible (importance across repeats or bootstrap).

---

## Unsupervised learning (rubric: Unsupervised learning — 5 pts) — **last section in report**

**Planned analyses:**

1. **PCA** on scaled numeric + optionally binary/dummy-expanded features (or PCA on numeric-only first, then extend). Scree plot; cumulative variance; **interpret leading loadings** (which variables drive PC1–PC3).
2. **Visualisation:** Scatter PC1 vs PC2 coloured by **`is_canceled`** (label used only for visualisation, not for fitting PCA) to see if cancellation separates in latent space.
3. **PC regression:** Predict `is_canceled` from first *q* PCs; compare AUC to full-feature models (shows how much signal is in low-rank structure).
4. **Clustering:** *k*-means or Gaussian mixture on a moderate number of PCs (e.g. 5–15). Choose *k* via elbow + silhouette. **Profile clusters:** mean `lead_time`, `adr`, segment mix, cancellation rate per cluster.
5. **Link to cancellation:** Chi-square or logistic with cluster label only; test whether clusters add value when appended to supervised models (nested CV or single holdout — document leakage avoidance: fit PCA/k-means **inside training fold only** if used as features).

This satisfies “at least one unsupervised analysis” and integrates findings into the narrative (segments as booking types with different risk).

---

## Limitations and criticisms (rubric: Limitations — 5 pts)

- **Leakage and timing:** Even after dropping obvious fields, some variables may reflect post-booking information (`country`, changes, waiting list) — acknowledge and show sensitivity analyses.
- **Confounding:** `deposit_type` and cancellation are entangled with **commercial policy**; coefficients are associational, not causal.
- **Duplicates and groups:** Deduplication changes the effective *n* and may remove legitimate repeat bookings; state assumption.
- **Generalisation:** Hotels and years are few; **time-based test** may show drop vs random split.
- **Class imbalance:** If using accuracy, complement with AUC and calibration.
- **Model limitations:** GAM additive; trees extrapolate poorly; k-NN curse of dimensionality after one-hot encoding (if used).
- **Unsupervised:** PCA assumes linear structure; k-means assumes spherical clusters — alternatives (kernel PCA, mixture models) as future work.

---

## Figures checklist for the 8-page report

1. **Data / EDA:** Cancellation rate by `lead_time` bins; by `deposit_type` or `market_segment`.
2. **Methods comparison:** Boxplots or line plots of **test AUC across repeated splits** for all models.
3. **Significance:** Table of corrected *p*-values for key pairwise comparisons (or forest plot of mean ΔAUC).
4. **Feature exploration:** Permutation importance bar chart; GAM smooths for top 2–3 variables.
5. **Unsupervised:** Scree + biplot or PC1–PC2 coloured by cancellation; cluster profiles (cancellation rate per cluster).
6. **Optional:** Learning curve justifying sample size; temporal holdout confusion matrix or calibration curve.

Each figure needs **one sentence** of takeaway in the caption.

---

## Suggested division of work (3–4 members)

| Role | Focus | Deliverables |
|------|--------|--------------|
| **Member A — Data & leakage** | Cleaning, dedup, leakage table, EDA figures, downsampling justification | Clean dataset notebook section; Data description draft |
| **Member B — Supervised models** | Implement logistic, GAM, RF, boosting; CV tuning; repeated splits | Model code; comparison tables; methods text |
| **Member C — Statistics & robustness** | Paired tests, corrected *t*-test, Holm; temporal split; calibration | Results subsection; significance table |
| **Member D — Features & unsupervised** | Importance, GAM shapes, PCA, clustering, cluster interpretation | Feature exploration + unsupervised sections; cluster figures |

If **three** people: merge C into B (statistics) and D (features + unsupervised).

---

## Execution order (notebook / code — for later)

1. Setup, load CSV, verify target column name **`is_canceled`**.
2. EDA, cleaning, leakage drops, feature engineering.
3. Downsampling + learning curve (if used).
4. Supervised: baselines → linear → GAM → ensembles; CV; repeated splits; tests.
5. Feature exploration.
6. Unsupervised: PCA → PC regression → clustering → link to cancellation.
7. Limitations and appendix references.

---

## References (optional for course scope)

- Antonio, N., Almeida, A., & Nunes, L. (2019). *Hotel booking demand datasets*. Data in Brief.
- Nadeau, C., & Bengio, Y. (2003). Inference for the generalization error. *Machine Learning*, 52(3), 239–281 (corrected resampled *t*-test).

---

*This plan is aligned with the ST5227 Fall 2026 project PDF (8-page report, executable code, rubric sections) and with EDA performed on the repository `hotel_bookings.csv`.*
