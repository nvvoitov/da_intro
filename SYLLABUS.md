# Introduction to Data Analysis
### Master's programme in Finance & Banking (joint with Gazprombank)

**Format:** 10 sessions (9 taught + 1 defence), 90–120 min each
**Prerequisites:** undergraduate econometrics (OLS, hypothesis testing), basic linear algebra, no Python required
**Core text:** James, Witten, Hastie, Tibshirani & Taylor, *An Introduction to Statistical Learning with Applications in Python* (ISLP), 2023 — free PDF at statlearning.com
**Language of instruction:** Russian; all code, identifiers and written deliverables in English

---

## 1. The thread

Every session in this course answers one part of a single question:

> **Which banks get into trouble, and can we see it coming?**

This is not a framing device. It is the applied problem that determines the syllabus. Failure prediction on a supervisory panel happens to exercise, in their natural order, almost every technique in ISLP — and it exercises them under the conditions that actually break analyses in industry: rare events, entity-correlated observations, time structure, regulatory constraints on model form, and a decision-maker who needs a threshold rather than a p-value.

Three levels of data sit under the question:

| Level | Unit of observation | Sessions |
|---|---|---|
| Bank | institution × quarter | 2–7 |
| Macro / policy | country or state × period | 8 |
| Market (optional) | security × day | bridge exercise, S5 |

The final deliverable is a single analytical report. Each homework is a section of it.

---

## 2. Learning outcomes

By the end of the course a student should be able to:

1. State whether a given business question is a **prediction** or an **inference** problem, and explain why the answer changes the method.
2. Design a validation protocol appropriate to panel data, and recognise leakage before a supervisor does.
3. Fit, tune and **calibrate** a classifier on a rare-event target, and convert its output into a decision under an asymmetric loss function.
4. Choose between an interpretable model and an accurate one, and defend the choice on grounds other than accuracy.
5. Estimate a treatment effect from observational data, state the identifying assumption in one sentence, and produce the diagnostic that would falsify it.
6. Produce a reproducible analysis that another person can re-run and a non-technical reader can act on.

Outcome 6 is graded as heavily as outcomes 3–5. This is deliberate.

---

## 3. Data

All course data is accessed programmatically. Nothing is distributed as a spreadsheet.

### Primary panel — FDIC BankFind

`https://banks.data.fdic.gov/api/` — no API key, JSON over HTTP, generous limits.

| Endpoint | Contents | Used in |
|---|---|---|
| `institutions` | registry: name, charter, state, active/inactive | 1–7 |
| `financials` | quarterly bank-level financials by `CERT` and `REPDTE` | 2–7 |
| `failures` | failure date, resolution type, cost to the insurance fund | 4, 7 |
| `sod` | Summary of Deposits — branch-level deposits | 6, 8 |

Students call it with `requests` in Session 1 — pagination, rate limits and field selection are part of the lesson. A pre-cleaned Parquet snapshot in `data/` is used from Session 2 onward so that a network outage never costs a seminar.

The panel in `data/banks_panel.parquet` is a **real FDIC pull** (2026-09-01), not a snapshot of the whole universe: it holds every institution that failed between 2001 and 2026 (586 of them) plus a random sample of 2 000 survivors — 152 152 bank-quarters, 2 586 banks, 21 MB. Sampling this way keeps the file small enough to read straight off GitHub in Colab while preserving every failure event the course depends on. The consequence to state out loud in class: **the failure base rate in this file (1.5 %) is higher than in the population (0.3 %)**, because survivors are under-sampled. Nothing else about the data is altered.

> **API note.** The endpoint moved: `banks.data.fdic.gov/api` now 301-redirects to **`api.fdic.gov/banks`**. The old host is what returns 403 from some networks; the new one works. Endpoints used: `financials` (quarterly, one request per quarter, `limit`/`offset` pagination), `institutions`, `failures`.
>
> **Two traps in the raw data**, both handled in the build and worth showing students: income-statement fields (`NETINC`, `NIM`, `NONIX`, `NTLNLS`) are **year-to-date cumulative** and must be differenced within each year to get quarterly flows; and the ratios have genuine pathological tails (banks with near-zero deposits or interest income), so every notebook filters explicitly rather than silently.

Why the US panel rather than the domestic one: it is a single well-documented API with a stable schema, ~4,000 active institutions, decades of history, and an unambiguous labelled event. Pedagogically it is the cleanest banking panel that exists.

### Domestic panel — Bank of Russia (optional track / final projects)

- Credit-institution web service (`cbr.ru/development/wsco/`) exposes forms **101** and **102** per institution, plus `GetFormsMaxDate` for forms 101, 102, 123, 134, 135.
- Bulk archives of form 101 turnover sheets are published as zipped DBF on the CBR reporting page.
- Forms **123 / 135** carry capital adequacy (Н1.0), the strongest single predictor in the domestic failure literature.
- Licence revocations 2013–2016 provide a dense event set.

Cost: SOAP, DBF, and a chart of accounts that shifts across the period. Ingestion is provided as a prepared Parquet plus an optional notebook showing the parsing; students are not asked to fight it in a seminar. Offer this track to students who want a domestically relevant final project — several will, and it is the version that impresses at a Gazprombank interview.

### Macro layer

| Source | Access | Series of interest |
|---|---|---|
| World Bank **GFDD** | `wbgapi` (switch off WDI to the Global Financial Development database) | bank Z-score, NPL ratio, concentration, net interest margin |
| BIS | `sdmx1` | credit-to-GDP gap, policy rates, property prices |
| IMF **FSI** | `sdmx1` | Financial Soundness Indicators |
| FRED | `fredapi` | H.8 bank credit, NFCI, term and credit spreads |

### Market layer (optional bridge exercise, S5)

`apimoex` — bank equity and bond series from MOEX ISS. Used once, for the Merton distance-to-default comparison.

---

## 4. Feature engineering scaffold: CAMELS

Students are not asked to invent features. They are given the supervisory taxonomy and asked to operationalise it — which is the actual skill.

| Dimension | Example ratios |
|---|---|
| **C**apital | equity / assets, tier-1 leverage, growth in equity |
| **A**sset quality | NPL / gross loans, loan-loss reserve coverage, net charge-off rate, CRE concentration |
| **M**anagement | cost-to-income, asset growth YoY, deviation of growth from peer median |
| **E**arnings | ROA, ROE, net interest margin, earnings volatility |
| **L**iquidity | liquid assets / assets, loans / deposits, brokered & wholesale funding share |
| **S**ensitivity | securities / assets, duration proxy, unrealised securities losses |

This gives a theory-driven ~25-feature design matrix instead of a 400-column kitchen sink, and gives every SHAP plot in Session 5 something to be interpreted *against*.

---

## 5. Sessions

Notation: **ISLP** = chapters to read before the session. **Lab** = what is coded live. **HW** = what is due at the start of the next session.

---

### S1 — What data analysis actually is

**ISLP** chapter 2, in full. This session *is* ISLP 2, taught on ISLP's own datasets before the course switches to the bank panel.

**Concepts.** $Y = f(X) + \varepsilon$ and what each symbol is. Prediction vs. inference, and why the same data answers them differently. Reducible and irreducible error. Parametric vs. non-parametric estimation of $f$. Interactions and powers as *bending the response surface* — the mechanism behind "the effect of $X_1$ depends on $X_2$". Flexibility vs. interpretability. Supervised vs. unsupervised. Regression vs. classification. MSE, overfitting, the bias-variance decomposition and degrees of freedom.

**Lab.** Built on the real ISLP tables (`Advertising`, `Income1`, `Income2`, `Auto`, `Heart`), so students can put the lecture next to the book:

- `Advertising` → the vocabulary, and the keystone demo: newspaper advertising is significant on its own ($t = 3.3$) and worthless once TV and radio are in the model ($p = 0.86$), because it correlates 0.35 with radio. One dataset, two questions, opposite answers.
- `Income2` → the linear plane (ISLP Fig. 2.3) against a thin-plate spline (Fig. 2.4), RSS 1394 vs 340, and then a spline with zero smoothing that passes through all 30 points — overfitting made visible before it is named.
- `Advertising` again → `TV × radio` twists the plane: the return on a TV dollar runs from 19 units at zero radio to 63 at radio = 40.
- `Auto` → PCA and *k*-means recover the engine classes without being told cylinders exist; then `mpg ~ horsepower` at polynomial degrees 1–12 gives the U-curve, with test MSE going from 20 to 18 892.
- `Heart` → classification, and why MSE is the wrong ruler for a qualitative response.

**Environment section (§0 of the notebook).** A from-scratch install guide for Windows and macOS — Python, VS Code, the Jupyter extension, `venv` — plus the table of the five things that actually break, and the standing advice to use Colab instead.

**Environment.** Google Colab, and nothing else. No installation, no terminal, no virtualenv — every notebook opens from a link and runs. Data is read straight off GitHub by URL. This cohort has never written Python; an hour lost to `pip` is an hour of the course lost. Students who want a local setup can have one, but it is their side project, not the course's.

**Readings.** Breiman (2001), *Statistical Modeling: The Two Cultures*, Statistical Science 16(3). Leek & Peng (2015), *What is the question?*, Science 347.

**HW0 (ungraded, pass/fail).** Working environment; load a dataset; one figure; 150 words on a claim your figure cannot support. *Note: HW00 still asks for a live FDIC pull, which L01 no longer demonstrates — realign it when the homework notebooks are rewritten.*

---

### S2 — Linear regression and the bias-variance trade-off

**ISLP** 2.2, 3
**Concepts.** OLS as estimator and OLS as predictor. Bias-variance decomposition. Why an economist's instinct (unbiasedness, t-statistics) and a predictor's instinct (test MSE) recommend different models from the same data. Standard errors when observations are clustered within banks.

**Lab.** Model ROA on CAMELS ratios. Present the result twice: once as a regression table with clustered SEs, once as a held-out RMSE. Discuss why the "best" specification differs.

**Readings.** Mullainathan & Spiess (2017), *Machine Learning: An Applied Econometric Approach*, JEP 31(2). This paper is the single most useful thing this cohort will read all term.

**HW1.** (a) ISLP 3.7 conceptual exercises 3 and 4. (b) Fit an ROA model; report an inference table and a test-set RMSE; 250 words on why the two criteria disagree and which you would use for which question.

---

### S3 — Validation and regularization  *(load-bearing session)*

**ISLP** 5, 6
**Concepts.** Validation set, k-fold, LOOCV, bootstrap. Ridge, lasso, elastic net; why collinear financial ratios make shrinkage necessary rather than decorative. Three species of leakage: **temporal** (future informs past), **entity** (same bank in train and test), **target** (post-event balance-sheet items encode the label).

**Lab — the centrepiece.** Students build a failure classifier with random k-fold and obtain an AUC near 0.99. They are congratulated. They then rebuild it with a time-based split and obtain something near 0.75. They spend the rest of the session diagnosing the gap. Do not warn them in advance.

**Readings.** Kaufman, Rosset & Perlich (2012), *Leakage in Data Mining*, KDD.

**HW2.** Reproduce both splits; quantify the gap; write the **validation protocol** you will use for the rest of the course, in one page, and justify each choice. This document is referenced in the final defence.

---

### S4 — Classification, imbalance and calibration

**ISLP** 4
**Concepts.** Logistic regression, LDA/QDA, naive Bayes. Base rates of 1–2% and what they do to accuracy, ROC and PR curves. **Calibration**: Brier score, reliability diagrams, Platt and isotonic rescaling. Threshold selection under an explicit asymmetric loss.

**Banking context.** WOE/IV scorecards and why banks still deploy logistic regression: model risk management, adverse-action explanation, supervisory review. ISLP's own `Default` dataset as the toy on-ramp before the real panel.

**Readings.** ISLP 4. Optional: Siddiqi, *Credit Risk Scorecards*, ch. 1–2.

**HW3.** Failure classifier under the S2 protocol; PR curve; reliability diagram before and after calibration; cost-optimal threshold with a stated loss function and a defence of the costs you assumed.

---

### S5 — Trees, ensembles and interpretation

**ISLP** 8
**Concepts.** CART, bagging, out-of-bag error, random forests, gradient boosting. LightGBM / CatBoost in practice. SHAP values and what they do and do not license you to say. **Monotonic constraints** — required in production credit models and a good illustration that a modelling choice can be a regulatory obligation.

**Debate.** Grinsztajn, Oyallon & Varoquaux (2022), *Why do tree-based models still outperform deep learning on tabular data?* — set against Rudin (2019), *Stop explaining black box machine learning models for high stakes decisions*, Nature MI 1.

**Optional bridge (offer, don't require).** Compute a Merton distance-to-default from MOEX equity volatility and form-101 liabilities for listed Russian banks; ask whether the structural measure adds anything over the ratio-based ML model. This is the exercise that connects the finance half of their degree to this course.

**Readings.** Lundberg & Lee (2017), *A Unified Approach to Interpreting Model Predictions*, NeurIPS. Grinsztajn et al. (2022). Rudin (2019).

**HW4.** GBDT under the identical S2 protocol; head-to-head against the S4 logistic model; SHAP summary; 300 words answering "would you deploy this, and what would the model risk function ask you?"

---

### S6 — Dimension reduction and clustering

**ISLP** 12
**Concepts.** PCA, k-means, hierarchical clustering. Choosing k honestly. Cluster **stability** via bootstrap resampling — the discipline that separates a taxonomy from a Rorschach blot.

**Application.** Bank business models. Roengpitya, Tarashev & Tsatsaronis (2014), *Bank business models*, BIS Quarterly Review, December. Students must **name** their clusters substantively (retail-funded, wholesale-funded, trading-oriented …) and defend the names.

**The honest finding.** Then test whether cluster membership improves the S5 model. It usually barely does. Sit with that: unsupervised structure is often descriptively valuable and predictively redundant, and saying so is a mark of a competent analyst.

**HW5.** Cluster; name; assess stability; test incremental predictive value; report the null result if you get one.

---

### S7 — Time to event

**ISLP** 11
**Concepts.** Censoring and why ignoring it biases everything. Kaplan–Meier. Cox proportional hazards and how to check the PH assumption. **Discrete-time hazard as logistic regression on person-period data** — the identity that lets an economist see S4 and S7 as the same object viewed twice.

**Banking context.** PD term structures; IFRS 9 lifetime expected credit loss; why a bank needs a hazard curve rather than a single probability.

**HW6.** KM survival curves by business-model cluster from S6; a Cox model on CAMELS covariates; 300 words on what the hazard framing tells you that the S4 classifier could not.

---

### S8 — Causal inference from observational data

**Not in ISLP** — reading pack provided.
**Concepts.** Potential outcomes and the fundamental problem. Difference-in-differences; the two-way fixed effects estimator and its recent troubles with staggered adoption and heterogeneous effects. Event-study plots and pre-trend diagnostics. Synthetic control.

**Applications** (pick one per team):
- US: the 2018 EGRRCPA move of the enhanced-supervision threshold from \$50bn to \$250bn — DiD, or an RDD in the assets dimension.
- Russia: the December 2014 increase of the deposit insurance limit from ₽700k to ₽1.4m — DiD on retail deposit growth, treatment intensity by pre-period share of deposits near the old cap.

**Contrast.** Fifteen minutes on what randomisation buys you for free, drawing on how an experiment is actually designed and defended in a commercial setting. The identifying assumptions look very different once you have had to argue for a real test design in front of a business owner.

**Readings.** Abadie (2021), *Using Synthetic Controls*, JEL 59(2). Goodman-Bacon (2021), JoE 225(2). Roth, Sant'Anna, Bilinski & Poe (2023), *What's trending in difference-in-differences?*, JoE 235(2).

**HW7.** A DiD estimate with an event-study figure, a parallel-trends diagnostic, and one honest paragraph naming the most plausible way your identification fails.

---

### S9 — Communicating analysis; reproducibility

**Concepts.** Plotly / Altair for interactivity; small multiples; when a dashboard is the wrong artefact and a single figure with a sentence is the right one. Cleveland & McGill's perceptual ranking as a design constraint rather than a style preference.

**Reproducibility.** Seeds, separating the data pull from the transform from the model, and the discipline that a notebook must run top-to-bottom in a fresh kernel. Discussed as professional practice — pinned environments, build targets, project structure — but demonstrated at the scale this cohort can actually carry: one notebook that a colleague can open and re-run.

**Writing.** The results-section grammar: *claim → evidence → uncertainty → limitation*. Every paragraph in the final report follows it.

**Readings.** Cleveland & McGill (1984), JASA 79(387). Wilson et al. (2017), *Good enough practices in scientific computing*, PLoS CB 13(6).

**HW8.** Take one figure from HW1–7 and rebuild it as a defensible final figure; write the 300-word results section that surrounds it.

---

### S10 — Defence

15 minutes per team: 8 minutes presentation, 7 minutes questions. Questions come from the instructor and from one assigned opposing team.

---

## 6. Final deliverable

One question about the banking sector, answered in:

- **Report** — 10 pages maximum, PDF, English. Structure: question, data, validation protocol, findings, limitations.
- **Notebooks** — each runs top-to-bottom in a fresh Colab kernel and regenerates every figure in the report.
- **Defence** — 8 minutes, plus questions.

The question is chosen in Session 3 and approved in Session 4. Approved questions from past cohorts are not reused.

Teams of two. Both members answer questions at the defence; a member who cannot explain a design choice made by their partner costs the team marks. This is stated in Session 1 and enforced.

---

## 7. Assessment

| Component | Weight |
|---|---|
| Homework 1–8 (best 7 of 8) | 40% |
| Final report | 35% |
| Defence | 25% |

### Report and defence rubric

| Criterion | Weight | What earns full marks |
|---|---|---|
| Validation design | 25% | Protocol matches the data structure; leakage anticipated, not discovered by the grader |
| Technical correctness | 20% | Estimator appropriate to the question; assumptions stated and checked |
| Substantive interpretation | 20% | Results connected to banking mechanisms, not just to coefficients |
| Honesty about uncertainty | 15% | Limitations named that a grader would otherwise have to find; null results reported |
| Reproducibility | 10% | Runs from clean clone |
| Communication | 10% | A risk officer could act on it without a follow-up meeting |

Note the weighting. A team with a well-validated model at AUC 0.74 and a candid limitations section outscores a team with 0.91 and an unexamined split. Say this out loud in Session 1, and then honour it in Session 3 when the leakage lab makes it concrete.

---

## 8. Policy on AI assistants

Students will use LLMs. The policy assumes it and is designed around it.

**Permitted, and encouraged:** explanation of concepts; debugging your own errors; explaining an unfamiliar library API; reviewing code you wrote; challenging your interpretation.

**Not permitted:** generating a solution to a graded task that you then submit; generating the prose of your report; generating results you did not run.

**Enforcement is by design, not by detection.** No tool is used to detect AI-written text. Instead:
- the defence is oral, and questions probe design choices rather than outputs;
- commit history is part of the submission, and a repository with one commit is a question to answer at the defence;
- HW2's validation protocol is referenced repeatedly, so a protocol the student cannot explain surfaces early.

A `CLAUDE.md` file is provided in the course repository. It configures Claude Code — and, pasted as a system prompt, any other assistant — to work as a tutor rather than a solution generator: it withholds final code for graded stubs and escalates hints only after you state your own attempt. Using it is in your interest. Deleting it is trivially possible and will not be detected; it will simply mean you arrive at the defence unable to answer questions about your own submission.

---

## 9. What this course deliberately omits

Stated in Session 1, so students know the boundary of what they have been taught:

- **Support vector machines** (ISLP 9) — omitted entirely; rarely competitive on tabular data and expensive in kernel intuition.
- **Non-linear methods** (ISLP 7) — one slide inside S2; splines and GAMs flagged as existing.
- **Multiple testing** (ISLP 13) — mentioned in S8 in the context of specification search.
- **Deep learning** (ISLP 10) — 20 minutes inside S5, framed as the tabular-data debate, not as a technique to be learned here.
- **Time-series forecasting** — not covered; flagged as the most important adjacent gap for this cohort.

The deep learning omission is a considered choice. One session produces students who can neither build a network nor understand one. Session 7 spends the same time on censoring, which they will use in their first year at a bank.
