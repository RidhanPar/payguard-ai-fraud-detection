# PayGuard Model Card

## Intended Use

PayGuard is a portfolio prototype for ranking card transactions by fraud probability and demonstrating explainable fraud-review workflows.

It must not be used as an autonomous blocking system or for real customer decisions without production validation, governance, monitoring, and human review.

## Data

- Primary dataset: public Kaggle credit-card fraud dataset.
- Demo dataset: synthetic records matching the public dataset schema.
- Features: anonymised PCA components plus transaction amount and time.
- Known limitation: anonymised features restrict business interpretation and actionability.

## Validation

- Test data remains separate from SMOTE-resampled training data.
- Scaling is fitted on training data only before transforming the untouched test split.
- Primary metrics: PR-AUC, ROC-AUC, precision, recall, F1, and fraud coverage at fixed analyst-review capacity.
- Published scores should only be treated as evidence when reproduced from the selected data source.

## Risks and Controls

- False positives can create customer friction and analyst workload.
- Fraud patterns and model calibration can drift.
- Demo-mode results do not represent production performance.
- Decisions should remain reviewable, explainable, monitored, and reversible.

## Monitoring Recommendation

Monitor score distribution, precision and recall at the operating threshold, PR-AUC, queue precision, fraud coverage at fixed review capacity, data drift, and analyst outcomes.
