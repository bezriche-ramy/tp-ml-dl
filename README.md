# TP2: K-Nearest Neighbors (KNN) — Report (answers extracted from tp2.c.py)

Date: November 2025

This file mirrors the TP2 questions and contains the responses taken from the `tp2.c.py` script in this repository.

## Part 3: Python Implementation on Security Dataset

### Question 1: What happens when you increase the value of K? How does data scaling affect KNN performance?

Effect of Increasing K:

- Increasing K -> smoother decision boundary and less sensitivity to noise; may underfit if K is too large.
- Small K (e.g., K=1): high variance, low bias, sensitive to outliers and noise; decision boundary is irregular and can overfit.
- Large K (e.g., K=5, K=7): lower variance, higher bias, decision boundary is smoother and local patterns may be missed.
- Optimal K should be selected via cross-validation.

Effect of Data Scaling:

- The script uses `StandardScaler` for the intrusion dataset. Scaling (standardization to mean=0, std=1) makes features comparable so no single feature dominates the distance calculation.
- For distance-based methods like KNN, scaling usually improves performance and fairness across features.


### Question 2: Try adding more samples or noise to the dataset, does MAE increase?

- Adding representative, clean samples typically reduces MAE (better generalization).
- Adding noisy samples or mislabeled data tends to increase MAE.
- Feature noise (small perturbations) causes small MAE increases; label noise causes larger increases; outliers can severely affect small-K models.

The script's note: "Adding noisy samples -> can increase MAE; more representative samples generally help."


### Question 3: Replace metric='euclidean' with metric='manhattan' in your model, does it change the results?

- The choice of metric can change which neighbors are selected and therefore change predictions.
- Manhattan (L1) can be more robust to outliers and can behave better in some feature distributions; Euclidean (L2) gives more weight to large differences due to squaring.
- The script tests both `manhattan` and `euclidean` for the synthetic intrusion dataset and notes that the metric choice depends on the features and distribution; always scale features.


## Part 4: KNN for Email Spam Detection

### Q3.1: What does each feature represent in this dataset?

- Feature 1: Number of Links — how many hyperlinks are present in the email; spam often contains many links to phishing sites or tracking URLs.
- Feature 2: Number of Spam Keywords — count of spammy trigger words like "free", "winner", "urgent"; higher counts indicate suspicious content.


### Q3.2: For K=1, 3, 5 → which prediction is correct compared to the true label?

New email: Links=2, SpamWords=1
True label (assumed in the script for evaluation): Spam (1)

Using the training data from `tp2.c.py` and the default Euclidean distance used in Part 4, the nearest neighbors to [2,1] are (calculated distances shown):

- Nearest neighbor (K=1): [1,2] (distance ≈ 1.414) — Label = Normal (0) → Predicted = 0 → MAE = 1 (incorrect)
- K=3 nearest labels: [0, 0, 1] (nearest three by distance) → majority = 0 → Predicted = 0 → MAE = 1 (incorrect)
- K=5 nearest labels: [0,0,1,0,1] → majority = 0 → Predicted = 0 → MAE = 1 (incorrect)

So for this particular toy training set and test point, all three tested K values produced the prediction "Normal (0)", which is incorrect relative to the assumed true label (Spam=1).


### Q3.3: Which K has the lowest MAE?

- For this single test sample, all tested K values (1, 3, 5) returned MAE = 1 (they all misclassified the test point). Therefore none has better MAE on this single example.
- (Reminder from the script: choose K using cross-validation on many samples rather than a single point.)


### Q3.4: Why can choosing a large value of K (such as K=5) in K-Nearest Neighbors lead to incorrect or less accurate predictions?

Main reasons (from the script's notes):

1. Over-smoothing — large K includes more, possibly distant, neighbors and removes local patterns.
2. Class imbalance — large K tends to favor the majority class and can drown out minority-class examples.
3. Irrelevant neighbors — far-away points can corrupt the local vote.
4. Curse of dimensionality — in high dimensions distances become less meaningful and large K worsens that effect.

The script also suggests practical rules: start around K ≈ √n, prefer odd K to avoid ties, and tune K via cross-validation.


### Q3.5: How could attackers try to bypass this detection system?

Attacker strategies the script lists (and common defenses):

- Adversarial feature manipulation: reduce visible links, use URL shorteners, embed links in images, obfuscate spam keywords (fr3e, homoglyphs), or add benign text to dilute spam signals.
- Mimic legitimate emails: copy newsletter formatting, add signatures, and legitimate-looking headers.
- Gradual drift: slowly change content to evade detection thresholds.
- Polymorphism: randomize content to avoid pattern repetition.
- Exploit model weaknesses: boundary probing, training-data poisoning, feature-space manipulation.

Defenses: use richer features (headers, sender reputation), ensembles, continuous retraining, anomaly detection, multi-layer systems, and user feedback.


## Summary (script's key takeaways)

1. K selection is a bias-variance trade-off: small K → noisy/overfit, large K → smooth/underfit.
2. Scale features for distance-based algorithms (the script uses `StandardScaler`).
3. Distance metric matters — choose based on data; try both Manhattan and Euclidean and compare with cross-validation.
4. Use cross-validation and multiple metrics (MAE, accuracy, F1) to select models and K.
5. KNN is a useful baseline for security tasks but is vulnerable to adversarial evasion — combine it with other methods in production.


## Notes about sources

- All answers and short notes were taken from the `tp2.c.py` script in this repository (the `answers_and_notes()` function and the comments/printouts in the parts).

---

