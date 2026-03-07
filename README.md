# Mini-AutoML for Tabular Data

## Project Description
This project is a simplified AutoML system for automatic binary classification on tabular datasets. The goal was to create a tool that removes the need for manual model picking. It automatically selects the best algorithms and combines them (ensembling) to make predictions more accurate and stable.

The choice of models is based on well-known industry benchmarks like OpenML-CC18.

## Key Features

* **Automatic Preprocessing:** The `fit` method handles everything—it fills in missing data, scales values, and converts text categories into numbers (one-hot encoding).
* **Smart Model Selection:** The system reads configurations from a JSON file and tests them using 5-fold cross-validation to see which ones perform best.
* **Large Model Portfolio:** Includes 50 optimized settings. It focuses on tree-based models (LightGBM, XGBoost, CatBoost, Random Forest), which are the best for table-like data. It also includes SVM and k-NN for better variety.
* **Handling Unbalanced Data:** All models are set up to handle datasets where one class has much more data than the other (`class_weight='balanced'`).
* **Ensembling:** Instead of using just one model, the system averages the results of the top 5 models (soft voting) to get a more reliable answer.

## How it Works

The main part of the project is the `MiniAutoML` class. It works just like a standard tool from the scikit-learn library:
* `__init__`: Sets up the system using a config file.
* `fit`: Prepares the data, ranks the models by their score, and trains the best ones.
* `predict`: Gives the final "Yes/No" classification.
* `predict_proba`: Shows the probability (percentage) for each class using the ensemble.

## Evaluation and Results

I tested the system on three different datasets: **Raisin** (image data), **Medical** (health risks), and **Income** (demographics).

**Main Findings:**
* **Stable Results:** Changing the random seed doesn't change the results much. This proves the system is reliable and consistent.
* **Adaptability:** Different datasets need different models. This shows that picking models automatically is much better than just using one favorite algorithm every time.
* **Ensembling Works:** The ensemble model (Top-5 average) is often as good as the best single model, but it is more "stable," meaning it’s less likely to make big mistakes on new data.

![Model selection frequency](images/model_selection_frequency.png)
*Fig 1: Which models were chosen most often for each dataset.*

![Best Model vs Ensemble](images/cv_vs_ensemble_raisins.png)
*Fig 2: Comparison between the single best model and the combined ensemble.*

![Stability across seeds](images/cv_stability_across_seeds.png)
*Fig 3: Final scores across different random seeds, showing consistent performance.*