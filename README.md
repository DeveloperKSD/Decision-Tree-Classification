# Decision Tree Classification (ID3)

A small ML lab experiment implementing decision tree classifiers using the ID3 (entropy-based) approach on two datasets.

## What's inside

- **Task 1:** Decision tree on a small custom dataset predicting infection status from symptoms (Fever, Cough, Breathing issues).
- **Task 2:** Decision tree classifying cancer diagnosis (malignant/benign) using the Breast Cancer Wisconsin dataset.

## Tech used

- Python
- scikit-learn (`DecisionTreeClassifier`)
- pandas
- matplotlib (for tree visualization)

## Approach

Both models use `criterion='entropy'`, which makes scikit-learn split nodes using **information gain** — the same core idea behind the ID3 algorithm, where the feature that most reduces uncertainty (entropy) about the target is chosen at each split.

- Categorical features (YES/NO) are encoded numerically with `LabelEncoder`.
- The cancer dataset is loaded directly via `sklearn.datasets.load_breast_cancer()` (Kaggle CSV also supported as an alternative).
- Models are evaluated using accuracy, and for the cancer dataset, a confusion matrix and classification report.
- Trees are visualized with `plot_tree` to see which features drive the splits.

## What I learnt

- How entropy and information gain decide which attribute to split on at each node.
- How a decision tree naturally handles both categorical and numerical data.
- Small datasets (like Task 1) can lead to overfitting — the tree can perfectly memorize training data without generalizing.
- Limiting `max_depth` helps control overfitting on larger, real-world datasets like the cancer data.
- Using `train_test_split` and accuracy on unseen test data gives a more honest picture of model performance than training accuracy alone.

## How to run

Open in Google Colab, run cells top to bottom. No external dataset download needed unless using the Kaggle CSV variant for Task 2.
