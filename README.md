## EC503 (Learning from Data) Project: Fall 2024

## Overview
This repository contains two Jupyter Notebook files that delve into theoretical and practical aspects of decision tree and random forest classifiers.

## Files
### 1. **DecisionTrees.ipynb**
- **Objective:** Understand and implement decision trees from a theoretical perspective.
- **Core Concepts Covered:**
  - **Entropy:** A measure of impurity or uncertainty in a dataset.
  - **Information Gain:** Reduction in entropy achieved after a split.
  - Recursive tree construction based on feature splits.
- **Practical Implementation:**
  - Manual calculation of entropy and information gain.
  - Recursive construction of a decision tree from scratch.
  - Evaluation of model performance on the Titanic dataset.

### 2. **RandomForest.ipynb**
- **Objective:** Explore the principles and advantages of ensemble learning with random forests.
- **Core Concepts Covered:**
  - **Ensemble Learning:** Combines predictions from multiple models to improve accuracy and reduce overfitting.
  - **Random Forests:** An ensemble of decision trees where each tree is trained on a bootstrap sample of the dataset and a random subset of features.
  - **Feature Importance:** Evaluates the contribution of each feature to the model’s predictions.
- **Practical Implementation:**
  - Training a random forest model using scikit-learn.
  - Tuning hyperparameters such as the number of trees (`n_estimators`) and maximum depth (`max_depth`).
  - Visualizing and interpreting feature importance.

## 3. **Theoretical Insights**
1. **Decision Trees:**
   - Effective for datasets with clear decision boundaries.
   - Prone to overfitting, especially with deep trees.
2. **Random Forests:**
   - Mitigates overfitting by averaging predictions from multiple trees.
   - Handles missing values and categorical variables effectively.
   - Provides robust performance on diverse datasets.

## 4. **Results**
- **Decision Tree Classifier:** Provides interpretability but may overfit.
  - Accuracy: ~81.7%.
- **Random Forest Classifier:** Balances accuracy and generalization with criteria as "entropy".
  - Training Accuracy: ~86.36%.
  - Testing Accuracy: ~83.58%.
