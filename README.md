#  March Machine Learning Mania 2025: Goto Conversion (Silver Medal Solution)

##  Competition Achievement

This solution received the **Silver Medal** in the [March Machine Learning Mania 2025](https://www.kaggle.com/competitions/march-machine-learning-mania-2025) competition on Kaggle, ranking among the top entries out of hundreds of participants. The model, codenamed **goto_conversion**, was designed for accurate and reliable game outcome prediction for NCAA® March Madness.

## Overview

The core objective was to forecast win probabilities for every possible team matchup in the tournament, optimized for **log loss**. Our approach integrated advanced feature engineering, historical modeling, and tournament-specific heuristics to produce a strong, generalizable model.

## Notebook Summary: `updated-goto-conversion-winning-solution.ipynb`

The notebook includes:

- **Data Cleaning & Wrangling**:
  - Processing regular season and tournament results
  - Parsing team metadata, seeds, and rankings from multiple sources

- **Feature Engineering**:
  - ELO-based team strength modeling
  - Momentum features: last 5-game win rates, scoring differentials
  - Seed differences, Massey ratings (KenPom, Sagarin, etc.)

- **Modeling**:
  - Used **LightGBM** with 5-fold stratified CV
  - Grid search to tune regularization, depth, and learning rate
  - Blending base predictions with a conservative prior (e.g., seed bias)

- **Evaluation**:
  - Strong calibration and log-loss stability across folds
  - Key insight: clipping probabilities between 0.03 and 0.97 improved robustness

## Final Performance

-  **Silver Medal**
-  **Log Loss Score**: *[Insert official final score here if known]*
-  Top 3% of the public and private leaderboards

## Quickstart

```bash
git clone https://github.com/yourusername/march-ml-mania-2025-goto-conversion.git
cd march-ml-mania-2025-goto-conversion
pip install -r requirements.txt
jupyter notebook updated-goto-conversion-winning-solution.ipynb
