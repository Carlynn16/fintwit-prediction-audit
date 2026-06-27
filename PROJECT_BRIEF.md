# PROJECT BRIEF — FinTwit Prediction Audit

**Working repo name:** `fintwit-prediction-audit` (rename if you prefer)

**One-line pitch:** An LLM-powered audit of ~18,000 stock predictions made by 51
financial influencers on X (2021–2025), asking a simple question with a rigorous
answer: *do they actually beat the market — and is anyone genuinely skilled, or
just lucky?*

**Deliverable type:** a reproducible, tested codebase + a written statistical report.
**No application or dashboard.** This is a descriptive + inferential statistical
analysis with a modelling component — not a product.

---

## 1. Why this project (positioning)

This is the portfolio "hero" piece. It is deliberately **different in substance**
from the existing statistical-report repos:

- It works on **unstructured text at scale** (NLP / LLM), not a clean tabular dataset.
- It sits at the intersection of **NLP + finance + statistical rigour** — a combination
  almost no other candidate brings.
- It includes a real **modelling** layer (skill-shrinkage model + predictive model).
- It is **honest-result friendly**: whatever the data says, the finding is interesting.
  This matches the established "negative/honest result" signature (cf. the carbon project).

The differentiation rests on content (novel NLP/LLM angle + rigour + honest finding),
not on packaging — the deliverable format (repo + Word report) stays consistent with
the rest of the portfolio.

**Provenance & framing:** this is a clean-room rebuild of a former paid consulting
engagement, fully anonymised and presented as a personal portfolio project (same
approach as the NASA-TLX rebuild). The original client is never named or referenced.

---

## 2. Research questions

**RQ1 — Accuracy & benchmark.** How accurate are FinTwit predictions directionally,
and do they beat a *passive* market benchmark (buy-and-hold of the named stock, and
the broad market via SPY) over the matched prediction horizon?

**RQ2 — Skill vs luck.** Across all 51 accounts, after correcting for multiple
comparisons, is *any individual account's* track record statistically distinguishable
from chance? Model true per-account skill with an empirical-Bayes / hierarchical
shrinkage estimator that accounts for small-sample noise.

**RQ3 — Signal value.** Do tweet-level signals — sentiment, trade type, horizon,
engagement, and the LLM's own self-reported `confidence` — carry any information
about whether a prediction comes true? Includes (a) a formal **calibration analysis**
of the LLM confidence score (a preliminary check already shows it is essentially
uncalibrated, r ≈ 0), and (b) a **predictive model** of `prediction_correct`.

---

## 3. Data

**Source:** 18,071 tweets from 51 financial-influencer accounts on X, 2021-01-07 →
2025-02-20. The raw tweet text and all metadata are already available locally, so
**no Twitter/X API access is required** — the project starts from the raw text and
rebuilds every derived field.

**Starting input (`data/tweets_raw.csv`), all genuinely raw fields:**

| Column | Meaning |
|---|---|
| `tweet_id`, `conversation_id`, `tweet_type` | identifiers; thread role (parent/reply) |
| `author` | account handle (pseudonymised — see ethics) |
| `text` | tweet body — the only NLP input |
| `created_at`, `created_date` | timestamp / date |
| `reply_to_tweet_id`, `reply_to_user` | reply structure |
| `likes`, `retweets`, `replies_count`, `views` | engagement |
| `author_followers`, `author_following`, `author_verified`, `author_blue_verified` | account metadata |

**Fields we REGENERATE from scratch (do not reuse the old outputs):**

- Ticker extraction: `tickers_mentioned`, `has_ticker`
- LLM classification: `time_horizon`, `trade_type`, `sentiment`, `confidence`
- Price validation: `prediction_date`, `price_change_pct`, `actual_return`,
  `prediction_correct`, `validated_ticker`, plus new **benchmark-adjusted** fields.

**Ethics / anonymisation (important):**
- Pseudonymise every account in the public repo (`account_01` … `account_51`). Keep
  the handle↔pseudonym mapping in a **gitignored** file. The public analysis is about
  the *population* of predictors, never about naming or shaming a specific individual.
- Ship the processed, pseudonymised dataset sufficient to reproduce the analysis.

---

## 4. Method (what makes it rigorous, not just "I used an LLM")

1. **Ticker extraction** — regex baseline (`$TICKER`) + LLM fallback for ambiguous
   cases; measured against a small hand-labelled sample (report precision/recall).
2. **LLM classification** — `gpt-4o-mini`, structured JSON output, **API key from
   environment variable**, retry/back-off on failures, and an **on-disk cache /
   checkpoint** keyed by tweet so re-runs are cheap and resumable. Each tweet →
   `time_horizon`, `trade_type`, `sentiment`, `confidence`.
3. **Price validation** — `yfinance`. Explicit, documented horizon windows (e.g. short
   ≈ 21 trading days, medium ≈ 63, long ≈ 126); price at tweet date vs horizon endpoint.
   A prediction is "directionally correct" if the price move sign matches the sentiment
   (neutral excluded). **Look-ahead-safe**, with explicit handling of missing data /
   delistings.
4. **Benchmark** — for every prediction also compute the return of (a) the broad market
   (SPY) and (b) buy-and-hold of the stock over the same window, and an **excess return
   vs SPY**. Turns "were they right?" into "did they beat the market?".
5. **Statistics & modelling**
   - Overall and segmented accuracy with **bootstrap / Wilson confidence intervals**.
   - Per-account skill: binomial test vs the relevant base rate, then **Benjamini-Hochberg
     FDR and Bonferroni** correction across the 51 accounts; plus an **empirical-Bayes /
     hierarchical shrinkage model** of true skill.
   - **Calibration** of LLM `confidence`: reliability curve + correlation with outcomes.
   - **Predictive model**: logistic regression and a tree-based model (gradient boosting)
     for `prediction_correct` on sentiment, trade type, horizon, engagement, confidence;
     honest evaluation (proper train/test or CV, AUC, calibration of the model itself).

---

## 5. Deliverables & repo structure

Standard structure:

```
figures/      scripts/      src/      tests/      notebooks/
data/         report/
README.md     PROJECT_BRIEF.md     requirements.txt     .gitignore
```

- `src/` — clean, importable modules (ingest, tickers, classify, validate, stats, models, viz).
- `tests/` — pytest, written alongside each module (LLM calls mocked; synthetic
  fixtures for the price/validation logic).
- `report/` — the Word report (**Times New Roman; body 12pt; level-1 headings 18pt;
  sub-headings 15pt; single space between number and title, e.g. "2.3 Overview";
  heading colour Blue Accent 1 Darker 25% / #2F5496**), exported to PDF. The `.docx`
  stays gitignored; only the PDF is committed.
- **The report is written incrementally — each phase appends its own section as it
  completes. The report is never assembled only at the end.**

---

## 6. Phase plan (build + test + report, phase by phase)

Each phase ends with: working code, its tests passing, and the corresponding report
section appended.

- **Phase 0 — Scaffold & data.** Repo skeleton, `requirements.txt`, `.gitignore`.
  Build `data/tweets_raw.csv` from source, pseudonymise accounts, clean/parse types,
  write the data dictionary. → Report: *1. Introduction*, *2. Data*.
- **Phase 1 — Ticker extraction.** Regex + LLM fallback; precision/recall on a labelled
  sample. → Report: *3. Ticker extraction*.
- **Phase 2 — LLM classification.** Cached, resumable, env-var key, tested with mocks.
  → Report: *4. Classification methodology*.
- **Phase 3 — Price validation & benchmark.** yfinance, horizon windows, excess return
  vs SPY, look-ahead safety. → Report: *5. Market validation*.
- **Phase 4 — RQ1.** Accuracy vs chance and vs benchmark, with CIs. → Report: *6. RQ1*.
- **Phase 5 — RQ2.** Per-account skill vs luck: multiple-testing correction + shrinkage
  model. → Report: *7. RQ2*.
- **Phase 6 — RQ3.** Signal value, confidence calibration, predictive model.
  → Report: *8. RQ3*.
- **Phase 7 — Polish.** Final report pass, README, figures, repo hygiene.

**Optional stretch (only if time allows, do not over-promise):**
GPT-vs-Claude classification comparison; tweet-deletion / posting-spike integrity checks.

---

## 7. Learning goals (this project teaches)

- Calling an LLM API for structured extraction (zero-shot, JSON output) at scale, with
  caching, retries, and cost control.
- Treating LLM outputs critically — calibration, why a "confidence" score is not a probability.
- Event-study-style validation against real market data, with a proper benchmark.
- The multiple-comparisons problem in a real "who is the best?" ranking, and shrinkage.
- Building and honestly evaluating a predictive model on a hard, noisy target.

---

## 8. Tech stack

Python; `pandas`, `numpy`; `openai` (classification); `yfinance` (prices);
`scipy` / `statsmodels` / `scikit-learn` (tests, regression, modelling; tree model via
`scikit-learn` or `lightgbm`); `matplotlib` (figures); `pytest` (tests);
`python-docx` (report).

---

## 9. Non-goals

- No financial advice; this is a descriptive/inferential audit.
- No application, dashboard, or deployment — the deliverable is a reproducible analysis
  plus a written report.
- No naming or ranking of identifiable individuals in public outputs.
