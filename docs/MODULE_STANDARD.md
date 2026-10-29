# DeepDraft.ai Module Standard

This document defines how every reference module in DeepDraft.ai should read and behave. It exists because the current modules share a useful skeleton (Use Cases, Fine-Tuning Recommendations, a `main()` demo) but differ widely in depth, accuracy, and style. A reader who opens any module should find the same learning arc in the same order, and every claim in the text should match what the code does.

The [repository review](review/REPOSITORY_REVIEW.md) explains the problems this standard solves. The [roadmap](../ROADMAP.md) schedules the rewrite.

---

## 1. The learning arc

Each module walks the reader through twelve sections, in this order. Sections 1 to 8 live in the module docstring. Sections 9 to 12 can live in the docstring or in the companion notebook, but the docstring must at least link to them.

| # | Section | Purpose | Length guide |
|---|---------|---------|--------------|
| 1 | **Summary** | One sentence: what the model does and for which task. | 1 to 2 lines |
| 2 | **When to use / when not to use** | Decision guidance, including the baseline the reader should beat first. | 4 to 8 bullets |
| 3 | **Intuition** | Plain-language explanation with no symbols. A smart analyst who has never seen the model should understand the core idea. | 3 to 6 sentences |
| 4 | **Model** | The math, with a notation table. Every symbol used later appears here. | As needed |
| 5 | **Algorithm** | Numbered pseudo-code. Each step names the function in this file that implements it. | 4 to 12 steps |
| 6 | **Assumptions and failure modes** | What must hold, and what you will observe when it does not. | 3 to 8 bullets |
| 7 | **Complexity** | Training time, inference time, and memory in big-O, plus a practical size limit. | 2 to 4 lines |
| 8 | **Hyperparameters** | A table: name, what it controls, default, typical range, tuning tip. | One row per knob |
| 9 | **Evaluation protocol** | How the demo splits data, which metric it reports and why, and which baseline it compares against. | 3 to 6 lines |
| 10 | **Expected output** | What the reader should see when `main()` runs on the bundled data (metric ranges, plot descriptions). | Short block |
| 11 | **Pitfalls** | The mistakes practitioners actually make with this model. | 3 to 8 bullets |
| 12 | **Related modules and references** | Cross-links inside DeepDraft.ai, then citations (paper, year) and library docs. | As needed |

### Rules for the text

1. **Claims must match code.** If the docstring lists a method, the file implements it or the text says "not implemented here; see `<library>`". The review found nine modules that describe methods they never implement.
2. **No private or project-specific references.** Use generic, public examples. Internal project names confuse outside readers.
3. **Cite sources.** Every non-obvious claim (an optimal acceptance rate, a rule of thumb, a benchmark result) gets a citation or a "rule of thumb" label.
4. **State defaults with the library version.** Write "scikit-learn 1.9 default: 100" rather than "default: 100", because defaults change.
5. **Explain outputs.** After every printed metric or plot, the text (or notebook) tells the reader how to read it.
6. **Prefer active voice and short sentences.** Write "The scaler centers each feature" rather than "Each feature is centered by the scaler."
7. **Use one notation table per module** and reuse the symbols from the shared glossary where possible (the glossary is planned in roadmap Phase 2).

### Rules for the code

1. **Map code to math.** Comments reference the algorithm step they implement, for example `# Step 3: E-step, compute responsibilities r_ik`.
2. **Never tune on the test set.** Choose hyperparameters, thresholds, and early-stopping points on a validation split or with cross-validation. Touch the test split once, at the end. `deepdraft.evaluation` provides `train_val_test_split`, `temporal_train_val_test_split`, and `rolling_origin_splits` for this.
3. **Never evaluate on training data** unless the output is explicitly labeled "in-sample".
4. **Seed everything** through one helper, `deepdraft.core.seed_everything`.
5. **No global warning suppression.** Silence a specific, understood warning with a targeted filter and a comment explaining why.
6. **Headless-safe plotting.** Save figures through a shared helper that respects an output directory and a `--no-show` flag.
7. **Fast mode.** `main(fast=True)` (or `DEEPDRAFT_FAST=1`) shrinks data and iterations so the smoke test finishes in seconds.
8. **Offline fallback.** Any demo that downloads data falls back to a synthetic or bundled sample when the network is unavailable.
9. **Configs as dataclasses.** Replace `# Placeholder` dictionaries with a typed `@dataclass` whose field docstrings carry the hyperparameter table from section 8.

---

## 2. Module docstring template

Copy this skeleton when writing or rewriting a module.

```python
"""
<Model Name>
<underline>

Summary:
    <One sentence: what it does, for which task.>

When to use:
    - <situation>
When not to use:
    - <situation, and the better alternative>
Baseline to beat first:
    - <simple model or heuristic, e.g. "majority class", "seasonal naive">

Intuition:
    <3 to 6 plain-language sentences.>

Model:
    <equations>

    Notation:
        x_i   feature vector of sample i, shape (d,)
        ...

Algorithm (functions in this file):
    1. <step>                                   -> fit_xxx()
    2. <step>                                   -> _update_xxx()
    ...

Assumptions and failure modes:
    - <assumption>: if violated, you will see <symptom>.

Complexity:
    Training: O(...)   Inference: O(...)   Memory: O(...)
    Practical limit: <e.g. "comfortable up to ~100k rows on a laptop">

Hyperparameters (library and version):
    | name | controls | default | typical range | tuning tip |

Evaluation protocol (this demo):
    <split strategy, metric, baseline>

Expected output (fast mode):
    <metric ranges and what each plot shows>

Pitfalls:
    - <mistake and how to avoid it>

Related modules:
    - <category/module.py>: <relationship>

References:
    - <Author (Year). Title. Venue.>
"""
```

---

## 3. Worked example: logistic regression

The current `classic/logistic_regression.py` docstring lists use cases, assumptions, and tuning tips, but it never explains the model, never shows the loss being minimized, and recommends a `multi_class` parameter that scikit-learn has since removed. Below is the same header rewritten to this standard. It is an illustration of tone and depth, not final copy.

```python
"""
Logistic Regression
===================

Summary:
    A linear classifier that models the probability of the positive class
    with the logistic (sigmoid) function and fits its weights by minimizing
    log-loss.

When to use:
    - You need calibrated-ish probabilities and coefficients you can explain.
    - The classes are roughly linearly separable after feature engineering.
    - You want a fast, strong baseline before trying tree ensembles.
When not to use:
    - Strong feature interactions or non-linear boundaries dominate
      (try gradient boosting; see ensemble/gradient_boosting.py).
    - Features vastly outnumber samples without regularization.
Baseline to beat first:
    - Majority-class predictor (accuracy) and a constant-probability
      predictor equal to the training prevalence (log-loss).

Intuition:
    The model computes a weighted sum of the features, a "score". Large
    positive scores mean "probably positive", large negative scores mean
    "probably negative". The sigmoid squashes the score into a probability
    between 0 and 1. Training adjusts the weights so the predicted
    probabilities match the observed labels as closely as possible, while a
    regularization term keeps the weights from growing without bound.

Model:
    p(y = 1 | x) = sigma(w^T x + b),   sigma(z) = 1 / (1 + e^(-z))

    Loss minimized (scikit-learn parameterization, L2 penalty):
        L(w, b) = C * sum_i [ -y_i log p_i - (1 - y_i) log(1 - p_i) ] + (1/2) ||w||^2

    Notation:
        x       feature vector, shape (d,)
        w, b    weights (d,) and intercept
        p_i     predicted probability for sample i
        C       inverse regularization strength (larger C = weaker penalty)

    The loss is convex, so every solver reaches the same optimum; solvers
    differ only in speed and in which penalties they support.

    Interpretation: on standardized features, exp(w_j) is the multiplicative
    change in the odds of the positive class for a one-standard-deviation
    increase in feature j, holding the others fixed.

Algorithm (functions in this file):
    1. Standardize features (fit the scaler on training data only)   -> build_pipeline()
    2. Minimize L(w, b) with L-BFGS                                    -> build_pipeline()
    3. Choose C by cross-validated log-loss on the training split      -> tune_hyperparameters()
    4. Pick a decision threshold on a validation split                 -> choose_threshold()
    5. Report metrics once on the untouched test split                 -> evaluate()

Assumptions and failure modes:
    - Linearity in the log-odds: if violated, residual plots by feature
      show curvature, and tree models win by a wide margin.
    - Limited multicollinearity: if violated, coefficients flip sign
      between CV folds even though accuracy stays stable.
    - Perfect separation: without regularization the weights diverge and
      the solver warns about convergence.

Complexity:
    Training: O(n * d) per solver iteration.   Inference: O(d) per sample.
    Practical limit: millions of rows with the 'saga' solver or SGDClassifier.

Hyperparameters (scikit-learn 1.9):
    | name         | controls                    | default | typical range   | tuning tip                               |
    | C            | inverse penalty strength    | 1.0     | 1e-3 to 1e3     | search on a log scale                    |
    | class_weight | per-class loss weights      | None    | None, balanced  | use 'balanced' when classes are skewed   |
    | solver       | optimization algorithm      | lbfgs   | lbfgs, saga     | 'saga' for large or sparse data          |
    | max_iter     | solver iteration cap        | 100     | 100 to 5000     | raise it if you see ConvergenceWarning   |

Evaluation protocol (this demo):
    Stratified 60/20/20 train/validation/test split. C is chosen by 5-fold
    CV on the training split, the threshold on the validation split, and
    ROC-AUC, log-loss, and F1 are reported once on the test split.

Expected output (fast mode):
    Test ROC-AUC around 0.90 to 0.95 on the synthetic data. The coefficient
    plot shows five clearly non-zero weights (the informative features) and
    smaller weights on the redundant and noise features.

Pitfalls:
    - Fitting the scaler on all data before splitting leaks test statistics.
    - Reading coefficients of unscaled features as importances.
    - Reporting accuracy alone on imbalanced data.
    - Moving the threshold on the test set.

Related modules:
    - classic/svm.py: max-margin linear and kernel classifiers.
    - ensemble/gradient_boosting.py: non-linear tabular baseline.
    - interpretability/shap_values.py: model-agnostic attributions.

References:
    - Hosmer, Lemeshow and Sturdivant (2013). Applied Logistic Regression, 3rd ed.
    - scikit-learn user guide, "Logistic regression".
"""
```

---

## 4. Review checklist for pull requests

A pull request that adds or rewrites a module should tick every box below. The checklist doubles as the content of the PR template planned in roadmap Phase 8.

- [ ] All twelve learning-arc sections are present (docs lint passes).
- [ ] Every method named in the docstring is implemented or explicitly marked "not implemented here".
- [ ] No hyperparameter, threshold, or early-stopping decision touches the test split.
- [ ] Metrics are reported on held-out data and labeled with the split name.
- [ ] `main(fast=True)` finishes in under 30 seconds on CPU.
- [ ] The smoke test and any reference-comparison tests pass.
- [ ] No global `warnings.filterwarnings("ignore")`.
- [ ] Figures go through the shared save helper; no bare `plt.show()`.
- [ ] Seeds come from `seed_everything`.
- [ ] References are cited; library defaults name the library version.
- [ ] No private project names or internal links.
