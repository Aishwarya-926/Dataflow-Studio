#!/usr/bin/env python3
"""
DataFlow Studio — Visual Data Science Workflow Builder
A Simulink-style drag-and-drop data science companion.
"""

import sys
import os
import json
import uuid
import pickle
import traceback
import threading
from typing import Optional, Dict, List, Any, Tuple

import numpy as np
import pandas as pd
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ─────────────────────────────────────────────────────────────────
#  THEME & CONSTANTS
# ─────────────────────────────────────────────────────────────────

THEME = {
    "bg_dark":       "#0D1117",
    "bg_mid":        "#161B22",
    "bg_light":      "#21262D",
    "bg_panel":      "#1C2128",
    "accent":        "#58A6FF",
    "accent2":       "#3FB950",
    "accent3":       "#F78166",
    "accent4":       "#D2A8FF",
    "accent5":       "#FFA657",
    "border":        "#30363D",
    "text":          "#E6EDF3",
    "text_dim":      "#8B949E",
    "text_muted":    "#484F58",
    "grid":          "#1E2530",
    "node_shadow":   "#000000",
    "wire":          "#58A6FF",
    "wire_active":   "#3FB950",
    "selection":     "#388BFD",
}

BLOCK_CATEGORIES = {
    "📂 Data I/O":    "#1F6FEB",
    "🔧 Preprocessing": "#238636",
    "🧬 Features":    "#6E40C9",
    "🤖 Models":      "#DA3633",
    "📊 Evaluation":  "#BF8700",
    "🔮 Prediction":  "#0D7377",
    "📈 Visualization": "#953800",
    "🧠 Deep Learning": "#8A2BE2",
    "⚙️ Advanced":      "#FF1493",
    
}

PORT_RADIUS = 6
PORT_HIT = 14
GRID_SIZE = 20
MIN_NODE_W = 180
NODE_H = 52
HEADER_H = 32


# ─────────────────────────────────────────────────────────────────
#  BLOCK DEFINITIONS
# ─────────────────────────────────────────────────────────────────

BLOCK_DEFS = {
    # ── Data I/O ──────────────────────────────────────────────────
    "Load CSV": {
        "category": "📂 Data I/O",
        "inputs": [],
        "outputs": ["DataFrame"],
        "params": {"file_path": {"type": "file", "label": "CSV File", "filter": "CSV (*.csv)"}},
        "description": "Load data from a CSV file",
        "icon": "📄",
    },
    "Load Parquet": {
        "category": "📂 Data I/O",
        "inputs": [],
        "outputs": ["DataFrame"],
        "params": {"file_path": {"type": "file", "label": "Parquet File", "filter": "Parquet (*.parquet)"}},
        "description": "Load data from a Parquet file",
        "icon": "📦",
    },
    "Save CSV": {
        "category": "📂 Data I/O",
        "inputs": ["DataFrame"],
        "outputs": [],
        "params": {"file_path": {"type": "save_file", "label": "Output File", "filter": "CSV (*.csv)"}},
        "description": "Save DataFrame to CSV",
        "icon": "💾",
    },
    "Sample Dataset": {
        "category": "📂 Data I/O",
        "inputs": [],
        "outputs": ["DataFrame"],
        "params": {
            "dataset": {"type": "combo", "label": "Dataset",
                        "options": ["Iris", "Boston Housing", "Breast Cancer", "Wine", "Diabetes", "Make Blobs", "Make Classification", "Make Regression"]}
        },
        "description": "Load a built-in sample dataset",
        "icon": "🗃️",
    },

    # ── Preprocessing ─────────────────────────────────────────────
    "Drop Missing": {
        "category": "🔧 Preprocessing",
        "inputs": ["DataFrame"],
        "outputs": ["DataFrame"],
        "params": {
            "axis": {"type": "combo", "label": "Drop Axis", "options": ["rows", "columns"]},
            "threshold": {"type": "float", "label": "Missing % Threshold", "default": 0.5, "min": 0.0, "max": 1.0},
        },
        "description": "Remove rows/columns with missing values",
        "icon": "🗑️",
    },
    "Impute Missing": {
        "category": "🔧 Preprocessing",
        "inputs": ["DataFrame"],
        "outputs": ["DataFrame"],
        "params": {
            "strategy": {"type": "combo", "label": "Strategy",
                         "options": ["mean", "median", "most_frequent", "constant"]},
            "fill_value": {"type": "str", "label": "Fill Value (if constant)", "default": "0"},
        },
        "description": "Fill missing values using imputation",
        "icon": "🩹",
    },
    "Standard Scaler": {
        "category": "🔧 Preprocessing",
        "inputs": ["DataFrame"],
        "outputs": ["DataFrame", "Scaler"],
        "params": {
            "columns": {"type": "str", "label": "Columns (comma-sep, blank=all numeric)", "default": ""},
        },
        "description": "Standardize features (zero mean, unit variance)",
        "icon": "⚖️",
    },
    "MinMax Scaler": {
        "category": "🔧 Preprocessing",
        "inputs": ["DataFrame"],
        "outputs": ["DataFrame", "Scaler"],
        "params": {
            "columns": {"type": "str", "label": "Columns (comma-sep, blank=all numeric)", "default": ""},
            "feature_min": {"type": "float", "label": "Min", "default": 0.0},
            "feature_max": {"type": "float", "label": "Max", "default": 1.0},
        },
        "description": "Scale features to a given range",
        "icon": "📏",
    },
    "Robust Scaler": {
        "category": "🔧 Preprocessing",
        "inputs": ["DataFrame"],
        "outputs": ["DataFrame", "Scaler"],
        "params": {
            "columns": {"type": "str", "label": "Columns (comma-sep, blank=all numeric)", "default": ""},
        },
        "description": "Scale using statistics robust to outliers",
        "icon": "🛡️",
    },
    "One-Hot Encode": {
        "category": "🔧 Preprocessing",
        "inputs": ["DataFrame"],
        "outputs": ["DataFrame"],
        "params": {
            "columns": {"type": "str", "label": "Columns (comma-sep, blank=all object)", "default": ""},
            "drop_first": {"type": "bool", "label": "Drop First", "default": False},
        },
        "description": "Encode categorical variables as one-hot vectors",
        "icon": "🔢",
    },
    "Label Encode": {
        "category": "🔧 Preprocessing",
        "inputs": ["DataFrame"],
        "outputs": ["DataFrame"],
        "params": {
            "columns": {"type": "str", "label": "Columns (comma-sep, blank=all object)", "default": ""},
        },
        "description": "Encode labels with integers",
        "icon": "🏷️",
    },
    "Remove Outliers": {
        "category": "🔧 Preprocessing",
        "inputs": ["DataFrame"],
        "outputs": ["DataFrame"],
        "params": {
            "method": {"type": "combo", "label": "Method", "options": ["IQR", "Z-Score"]},
            "threshold": {"type": "float", "label": "Threshold", "default": 3.0},
        },
        "description": "Detect and remove outlier rows",
        "icon": "🎯",
    },
    "Log Transform": {
        "category": "🔧 Preprocessing",
        "inputs": ["DataFrame"],
        "outputs": ["DataFrame"],
        "params": {
            "columns": {"type": "str", "label": "Columns (comma-sep, blank=all numeric)", "default": ""},
            "base": {"type": "combo", "label": "Base", "options": ["natural (e)", "log2", "log10"]},
        },
        "description": "Apply logarithmic transformation",
        "icon": "📐",
    },
    "Select Features": {
        "category": "🔧 Preprocessing",
        "inputs": ["DataFrame"],
        "outputs": ["Features", "Target"],
        "params": {
            "target_col": {"type": "str", "label": "Target Column", "default": "target"},
            "feature_cols": {"type": "str", "label": "Feature Cols (blank=all others)", "default": ""},
        },
        "description": "Split DataFrame into features X and target y",
        "icon": "✂️",
    },
    "Train/Test Split": {
        "category": "🔧 Preprocessing",
        "inputs": ["Features", "Target"],
        "outputs": ["X_train", "X_test", "y_train", "y_test"],
        "params": {
            "test_size": {"type": "float", "label": "Test Size", "default": 0.2, "min": 0.05, "max": 0.5},
            "random_state": {"type": "int", "label": "Random State", "default": 42},
            "stratify": {"type": "bool", "label": "Stratify (classification)", "default": False},
        },
        "description": "Split data into training and testing sets",
        "icon": "🔀",
    },
    "PCA": {
        "category": "🔧 Preprocessing",
        "inputs": ["DataFrame"],
        "outputs": ["DataFrame"],
        "params": {
            "n_components": {"type": "int", "label": "Components", "default": 2, "min": 1},
        },
        "description": "Principal Component Analysis dimensionality reduction",
        "icon": "🌀",
    },

    # ── Feature Extraction ────────────────────────────────────────
    "TF-IDF": {
        "category": "🧬 Features",
        "inputs": ["DataFrame"],
        "outputs": ["DataFrame"],
        "params": {
            "text_column": {"type": "str", "label": "Text Column", "default": "text"},
            "max_features": {"type": "int", "label": "Max Features", "default": 1000},
            "max_df": {"type": "float", "label": "Max DF", "default": 0.95},
            "min_df": {"type": "int", "label": "Min DF", "default": 1},
        },
        "description": "Term Frequency-Inverse Document Frequency vectorizer",
        "icon": "📝",
    },
    "Count Vectorizer": {
        "category": "🧬 Features",
        "inputs": ["DataFrame"],
        "outputs": ["DataFrame"],
        "params": {
            "text_column": {"type": "str", "label": "Text Column", "default": "text"},
            "max_features": {"type": "int", "label": "Max Features", "default": 1000},
        },
        "description": "Bag-of-words text feature extraction",
        "icon": "📊",
    },
    "Time Series Features": {
        "category": "🧬 Features",
        "inputs": ["DataFrame"],
        "outputs": ["DataFrame"],
        "params": {
            "date_column": {"type": "str", "label": "Date Column", "default": "date"},
        },
        "description": "Extract time-based features from datetime column",
        "icon": "🕐",
    },

    # ── Models: Regression ────────────────────────────────────────
    "Linear Regression": {
        "category": "🤖 Models",
        "inputs": ["X_train", "y_train"],
        "outputs": ["Model"],
        "params": {
            "fit_intercept": {"type": "bool", "label": "Fit Intercept", "default": True},
        },
        "description": "Ordinary least squares linear regression",
        "icon": "📉",
        "task": "regression",
    },
    "Ridge Regression": {
        "category": "🤖 Models",
        "inputs": ["X_train", "y_train"],
        "outputs": ["Model"],
        "params": {
            "alpha": {"type": "float", "label": "Alpha (L2)", "default": 1.0},
        },
        "description": "L2-regularized linear regression",
        "icon": "🏔️",
        "task": "regression",
    },
    "Lasso Regression": {
        "category": "🤖 Models",
        "inputs": ["X_train", "y_train"],
        "outputs": ["Model"],
        "params": {
            "alpha": {"type": "float", "label": "Alpha (L1)", "default": 1.0},
        },
        "description": "L1-regularized linear regression",
        "icon": "🔱",
        "task": "regression",
    },
    "Random Forest Regressor": {
        "category": "🤖 Models",
        "inputs": ["X_train", "y_train"],
        "outputs": ["Model"],
        "params": {
            "n_estimators": {"type": "int", "label": "Trees", "default": 100},
            "max_depth": {"type": "int", "label": "Max Depth (0=None)", "default": 0},
            "random_state": {"type": "int", "label": "Random State", "default": 42},
        },
        "description": "Ensemble of decision tree regressors",
        "icon": "🌲",
        "task": "regression",
    },
    "Gradient Boosting Regressor": {
        "category": "🤖 Models",
        "inputs": ["X_train", "y_train"],
        "outputs": ["Model"],
        "params": {
            "n_estimators": {"type": "int", "label": "Estimators", "default": 100},
            "learning_rate": {"type": "float", "label": "Learning Rate", "default": 0.1},
            "max_depth": {"type": "int", "label": "Max Depth", "default": 3},
        },
        "description": "Gradient boosted trees for regression",
        "icon": "🚀",
        "task": "regression",
    },
    "SVR": {
        "category": "🤖 Models",
        "inputs": ["X_train", "y_train"],
        "outputs": ["Model"],
        "params": {
            "kernel": {"type": "combo", "label": "Kernel", "options": ["rbf", "linear", "poly", "sigmoid"]},
            "C": {"type": "float", "label": "C", "default": 1.0},
            "epsilon": {"type": "float", "label": "Epsilon", "default": 0.1},
        },
        "description": "Support Vector Regression",
        "icon": "⚡",
        "task": "regression",
    },

    # ── Models: Classification ────────────────────────────────────
    "Logistic Regression": {
        "category": "🤖 Models",
        "inputs": ["X_train", "y_train"],
        "outputs": ["Model"],
        "params": {
            "C": {"type": "float", "label": "Inverse Regularization (C)", "default": 1.0},
            "max_iter": {"type": "int", "label": "Max Iterations", "default": 1000},
            "solver": {"type": "combo", "label": "Solver", "options": ["lbfgs", "liblinear", "saga", "sag"]},
        },
        "description": "Linear model for binary/multiclass classification",
        "icon": "🧮",
        "task": "classification",
    },
    "Random Forest Classifier": {
        "category": "🤖 Models",
        "inputs": ["X_train", "y_train"],
        "outputs": ["Model"],
        "params": {
            "n_estimators": {"type": "int", "label": "Trees", "default": 100},
            "max_depth": {"type": "int", "label": "Max Depth (0=None)", "default": 0},
            "random_state": {"type": "int", "label": "Random State", "default": 42},
        },
        "description": "Ensemble of decision tree classifiers",
        "icon": "🌳",
        "task": "classification",
    },
    "SVM Classifier": {
        "category": "🤖 Models",
        "inputs": ["X_train", "y_train"],
        "outputs": ["Model"],
        "params": {
            "kernel": {"type": "combo", "label": "Kernel", "options": ["rbf", "linear", "poly"]},
            "C": {"type": "float", "label": "C", "default": 1.0},
        },
        "description": "Support Vector Machine classifier",
        "icon": "🎪",
        "task": "classification",
    },
    "KNN Classifier": {
        "category": "🤖 Models",
        "inputs": ["X_train", "y_train"],
        "outputs": ["Model"],
        "params": {
            "n_neighbors": {"type": "int", "label": "K Neighbors", "default": 5},
            "weights": {"type": "combo", "label": "Weights", "options": ["uniform", "distance"]},
        },
        "description": "K-Nearest Neighbors classifier",
        "icon": "🔍",
        "task": "classification",
    },
    "Gradient Boosting Classifier": {
        "category": "🤖 Models",
        "inputs": ["X_train", "y_train"],
        "outputs": ["Model"],
        "params": {
            "n_estimators": {"type": "int", "label": "Estimators", "default": 100},
            "learning_rate": {"type": "float", "label": "Learning Rate", "default": 0.1},
            "max_depth": {"type": "int", "label": "Max Depth", "default": 3},
        },
        "description": "Gradient boosted trees for classification",
        "icon": "💥",
        "task": "classification",
    },
    "Decision Tree Classifier": {
        "category": "🤖 Models",
        "inputs": ["X_train", "y_train"],
        "outputs": ["Model"],
        "params": {
            "max_depth": {"type": "int", "label": "Max Depth (0=None)", "default": 0},
            "criterion": {"type": "combo", "label": "Criterion", "options": ["gini", "entropy"]},
        },
        "description": "Decision tree for classification",
        "icon": "🌿",
        "task": "classification",
    },
    "Naive Bayes": {
        "category": "🤖 Models",
        "inputs": ["X_train", "y_train"],
        "outputs": ["Model"],
        "params": {},
        "description": "Gaussian Naive Bayes classifier",
        "icon": "🎲",
        "task": "classification",
    },

    # ── Models: Clustering ────────────────────────────────────────
    "K-Means": {
        "category": "🤖 Models",
        "inputs": ["DataFrame"],
        "outputs": ["DataFrame", "Model"],
        "params": {
            "n_clusters": {"type": "int", "label": "Clusters", "default": 3},
            "random_state": {"type": "int", "label": "Random State", "default": 42},
            "n_init": {"type": "int", "label": "N Init", "default": 10},
        },
        "description": "K-Means clustering",
        "icon": "🎯",
        "task": "clustering",
    },
    "DBSCAN": {
        "category": "🤖 Models",
        "inputs": ["DataFrame"],
        "outputs": ["DataFrame", "Model"],
        "params": {
            "eps": {"type": "float", "label": "Epsilon", "default": 0.5},
            "min_samples": {"type": "int", "label": "Min Samples", "default": 5},
        },
        "description": "Density-based spatial clustering",
        "icon": "🌌",
        "task": "clustering",
    },
    "Isolation Forest": {
        "category": "🤖 Models",
        "inputs": ["DataFrame"],
        "outputs": ["DataFrame", "Model"],
        "params": {
            "contamination": {"type": "float", "label": "Contamination", "default": 0.1},
            "n_estimators": {"type": "int", "label": "Estimators", "default": 100},
        },
        "description": "Anomaly detection via isolation forest",
        "icon": "🚨",
        "task": "anomaly",
    },
    # ── Deep Learning ─────────────────────────────────────────────
    "MLP Classifier": {
        "category": "🧠 Deep Learning",
        "inputs": ["X_train", "y_train"],
        "outputs": ["Model"],
        "params": {
            "hidden_layers": {"type": "str", "label": "Hidden Layers (comma-sep)", "default": "100,50"},
            "max_iter": {"type": "int", "label": "Max Iterations", "default": 200},
            "learning_rate": {"type": "float", "label": "Learning Rate Init", "default": 0.001},
        },
        "description": "Multi-Layer Perceptron (Deep Neural Network)",
        "icon": "🧠",
        "task": "classification",
    },

    # ── Advanced Tuning & Ensembling ──────────────────────────────
    "Grid Search CV": {
        "category": "⚙️ Advanced",
        "inputs": ["Model", "Features", "Target"],
        "outputs": ["Model", "Metrics"],
        "params": {
            "param_grid": {"type": "str", "label": "Grid (JSON format)", "default": '{"max_depth": [3, 5, 10]}'},
            "cv": {"type": "int", "label": "CV Folds", "default": 3},
        },
        "description": "Automated hyperparameter tuning to find the best model",
        "icon": "🎛️",
    },
    "Voting Classifier": {
        "category": "⚙️ Advanced",
        "inputs": ["Model 1", "Model 2", "X_train", "y_train"],
        "outputs": ["Model"],
        "params": {
            "voting": {"type": "combo", "label": "Voting Type", "options": ["hard", "soft"]},
        },
        "description": "Combine two different models via a voting ensemble",
        "icon": "🤝",
    },

    # ── Evaluation ────────────────────────────────────────────────
    "Regression Metrics": {
        "category": "📊 Evaluation",
        "inputs": ["Model", "X_test", "y_test"],
        "outputs": ["Metrics"],
        "params": {},
        "description": "Compute RMSE, MAE, R² for regression",
        "icon": "📐",
    },
    "Classification Metrics": {
        "category": "📊 Evaluation",
        "inputs": ["Model", "X_test", "y_test"],
        "outputs": ["Metrics"],
        "params": {},
        "description": "Compute accuracy, F1, precision, recall, ROC AUC",
        "icon": "🏆",
    },
    "Cross Validation": {
        "category": "📊 Evaluation",
        "inputs": ["Model", "Features", "Target"],
        "outputs": ["Metrics"],
        "params": {
            "cv": {"type": "int", "label": "Folds", "default": 5},
            "scoring": {"type": "combo", "label": "Scoring",
                        "options": ["accuracy", "f1_weighted", "r2", "neg_root_mean_squared_error"]},
        },
        "description": "K-fold cross-validation evaluation",
        "icon": "🔄",
    },
    "Feature Importance": {
        "category": "📊 Evaluation",
        "inputs": ["Model", "Features"],
        "outputs": ["Metrics"],
        "params": {"top_n": {"type": "int", "label": "Top N Features", "default": 15}},
        "description": "Plot feature importance from tree-based model",
        "icon": "⭐",
    },

    # ── Prediction ────────────────────────────────────────────────
    "Predict": {
        "category": "🔮 Prediction",
        "inputs": ["Model", "DataFrame"],
        "outputs": ["DataFrame"],
        "params": {
            "output_col": {"type": "str", "label": "Output Column Name", "default": "prediction"},
        },
        "description": "Apply trained model to new data",
        "icon": "🔮",
    },
    "Predict Proba": {
        "category": "🔮 Prediction",
        "inputs": ["Model", "DataFrame"],
        "outputs": ["DataFrame"],
        "params": {
            "output_prefix": {"type": "str", "label": "Column Prefix", "default": "prob_"},
        },
        "description": "Predict class probabilities (classifiers)",
        "icon": "🎰",
    },

    # ── Visualization ─────────────────────────────────────────────
    "Histogram": {
        "category": "📈 Visualization",
        "inputs": ["DataFrame"],
        "outputs": [],
        "params": {
            "column": {"type": "str", "label": "Column (blank=all numeric)", "default": ""},
            "bins": {"type": "int", "label": "Bins", "default": 30},
        },
        "description": "Plot histogram of feature distributions",
        "icon": "📊",
    },
    "Scatter Plot": {
        "category": "📈 Visualization",
        "inputs": ["DataFrame"],
        "outputs": [],
        "params": {
            "x_col": {"type": "str", "label": "X Column", "default": "x"},
            "y_col": {"type": "str", "label": "Y Column", "default": "y"},
            "color_col": {"type": "str", "label": "Color Column (optional)", "default": ""},
        },
        "description": "Scatter plot of two features",
        "icon": "🔵",
    },
    "Correlation Heatmap": {
        "category": "📈 Visualization",
        "inputs": ["DataFrame"],
        "outputs": [],
        "params": {},
        "description": "Heatmap of feature correlations",
        "icon": "🌡️",
    },
    "Confusion Matrix": {
        "category": "📈 Visualization",
        "inputs": ["Model", "X_test", "y_test"],
        "outputs": [],
        "params": {},
        "description": "Plot confusion matrix for classifiers",
        "icon": "🧩",
    },
    "ROC Curve": {
        "category": "📈 Visualization",
        "inputs": ["Model", "X_test", "y_test"],
        "outputs": [],
        "params": {},
        "description": "Plot ROC curve and compute AUC",
        "icon": "📡",
    },
    "Residual Plot": {
        "category": "📈 Visualization",
        "inputs": ["Model", "X_test", "y_test"],
        "outputs": [],
        "params": {},
        "description": "Plot residuals for regression diagnostics",
        "icon": "〰️",
    },
    "Box Plot": {
        "category": "📈 Visualization",
        "inputs": ["DataFrame"],
        "outputs": [],
        "params": {
            "columns": {"type": "str", "label": "Columns (blank=all numeric)", "default": ""},
        },
        "description": "Box plots showing distribution statistics",
        "icon": "📦",
    },
    "Show Metrics": {
        "category": "📈 Visualization",
        "inputs": ["Metrics"],
        "outputs": [],
        "params": {},
        "description": "Display evaluation metrics as a table",
        "icon": "📋",
    },
    "Data Summary": {
        "category": "📈 Visualization",
        "inputs": ["DataFrame"],
        "outputs": [],
        "params": {},
        "description": "Show descriptive statistics of the DataFrame",
        "icon": "📜",
    },
}


# ─────────────────────────────────────────────────────────────────
#  EXECUTION ENGINE
# ─────────────────────────────────────────────────────────────────

class ExecutionEngine:
    """Executes block logic given its name and parameters."""

    def run_block(self, block_name: str, params: dict, inputs: dict) -> dict:
        fn = getattr(self, f"_run_{block_name.lower().replace(' ', '_').replace('-', '_').replace('/', '_')}", None)
        if fn is None:
            raise NotImplementedError(f"Block '{block_name}' execution not implemented.")
        return fn(params, inputs)

    # ── Data I/O ──────────────────────────────────────────────────
    def _run_load_csv(self, p, inp):
        path = p.get("file_path", "")
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"CSV file not found: {path}")
        df = pd.read_csv(path)
        return {"DataFrame": df}

    def _run_load_parquet(self, p, inp):
        path = p.get("file_path", "")
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"Parquet file not found: {path}")
        df = pd.read_parquet(path)
        return {"DataFrame": df}

    def _run_save_csv(self, p, inp):
        df = inp.get("DataFrame")
        if df is None: raise ValueError("No DataFrame input")
        path = p.get("file_path", "output.csv")
        df.to_csv(path, index=False)
        return {"_info": f"Saved {len(df)} rows to {path}"}

    def _run_sample_dataset(self, p, inp):
        from sklearn import datasets
        name = p.get("dataset", "Iris")
        if name == "Iris":
            d = datasets.load_iris(as_frame=True)
            df = d.frame
        elif name == "Boston Housing":
            # Boston removed from sklearn; use California housing instead
            d = datasets.fetch_california_housing(as_frame=True)
            df = d.frame
        elif name == "Breast Cancer":
            d = datasets.load_breast_cancer(as_frame=True)
            df = d.frame
        elif name == "Wine":
            d = datasets.load_wine(as_frame=True)
            df = d.frame
        elif name == "Diabetes":
            d = datasets.load_diabetes(as_frame=True)
            df = d.frame
        elif name == "Make Blobs":
            X, y = datasets.make_blobs(n_samples=300, centers=4, random_state=42)
            df = pd.DataFrame(X, columns=["x0", "x1"])
            df["target"] = y
        elif name == "Make Classification":
            X, y = datasets.make_classification(n_samples=500, n_features=10, random_state=42)
            df = pd.DataFrame(X, columns=[f"f{i}" for i in range(10)])
            df["target"] = y
        else:  # Make Regression
            X, y = datasets.make_regression(n_samples=500, n_features=8, noise=20, random_state=42)
            df = pd.DataFrame(X, columns=[f"f{i}" for i in range(8)])
            df["target"] = y
        return {"DataFrame": df}

    # ── Preprocessing ─────────────────────────────────────────────
    def _run_drop_missing(self, p, inp):
        df = inp["DataFrame"].copy()
        axis = 0 if p.get("axis", "rows") == "rows" else 1
        thresh = float(p.get("threshold", 0.5))
        if axis == 0:
            min_count = int((1 - thresh) * df.shape[1])
            df = df.dropna(axis=0, thresh=min_count)
        else:
            min_count = int((1 - thresh) * df.shape[0])
            df = df.dropna(axis=1, thresh=min_count)
        return {"DataFrame": df}

    def _run_impute_missing(self, p, inp):
        from sklearn.impute import SimpleImputer
        df = inp["DataFrame"].copy()
        strategy = p.get("strategy", "mean")
        fill_value = p.get("fill_value", "0")
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        if num_cols:
            kw = {"strategy": strategy}
            if strategy == "constant":
                try: kw["fill_value"] = float(fill_value)
                except: kw["fill_value"] = fill_value
            imp = SimpleImputer(**kw)
            df[num_cols] = imp.fit_transform(df[num_cols])
        return {"DataFrame": df}

    def _scale_df(self, df, cols, scaler_obj):
        from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
        if not cols:
            cols = df.select_dtypes(include=np.number).columns.tolist()
        df = df.copy()
        df[cols] = scaler_obj.fit_transform(df[cols])
        return df, scaler_obj

    def _run_standard_scaler(self, p, inp):
        from sklearn.preprocessing import StandardScaler
        df = inp["DataFrame"]
        cols = [c.strip() for c in p.get("columns", "").split(",") if c.strip()]
        df_out, scaler = self._scale_df(df, cols, StandardScaler())
        return {"DataFrame": df_out, "Scaler": scaler}

    def _run_minmax_scaler(self, p, inp):
        from sklearn.preprocessing import MinMaxScaler
        df = inp["DataFrame"]
        cols = [c.strip() for c in p.get("columns", "").split(",") if c.strip()]
        fr = (float(p.get("feature_min", 0.0)), float(p.get("feature_max", 1.0)))
        df_out, scaler = self._scale_df(df, cols, MinMaxScaler(feature_range=fr))
        return {"DataFrame": df_out, "Scaler": scaler}

    def _run_robust_scaler(self, p, inp):
        from sklearn.preprocessing import RobustScaler
        df = inp["DataFrame"]
        cols = [c.strip() for c in p.get("columns", "").split(",") if c.strip()]
        df_out, scaler = self._scale_df(df, cols, RobustScaler())
        return {"DataFrame": df_out, "Scaler": scaler}

    def _run_one_hot_encode(self, p, inp):
        df = inp["DataFrame"].copy()
        cols = [c.strip() for c in p.get("columns", "").split(",") if c.strip()]
        if not cols:
            cols = df.select_dtypes(include="object").columns.tolist()
        drop = p.get("drop_first", False)
        if cols:
            df = pd.get_dummies(df, columns=cols, drop_first=drop)
        return {"DataFrame": df}

    def _run_label_encode(self, p, inp):
        from sklearn.preprocessing import LabelEncoder
        df = inp["DataFrame"].copy()
        cols = [c.strip() for c in p.get("columns", "").split(",") if c.strip()]
        if not cols:
            cols = df.select_dtypes(include="object").columns.tolist()
        for c in cols:
            df[c] = LabelEncoder().fit_transform(df[c].astype(str))
        return {"DataFrame": df}

    def _run_remove_outliers(self, p, inp):
        df = inp["DataFrame"].copy()
        method = p.get("method", "IQR")
        thresh = float(p.get("threshold", 3.0))
        num_cols = df.select_dtypes(include=np.number).columns
        if method == "IQR":
            mask = pd.Series([True] * len(df))
            for c in num_cols:
                Q1, Q3 = df[c].quantile(0.25), df[c].quantile(0.75)
                IQR = Q3 - Q1
                mask &= (df[c] >= Q1 - thresh * IQR) & (df[c] <= Q3 + thresh * IQR)
            df = df[mask]
        else:  # Z-Score
            from scipy import stats
            z = np.abs(stats.zscore(df[num_cols].fillna(df[num_cols].mean())))
            df = df[(z < thresh).all(axis=1)]
        return {"DataFrame": df}

    def _run_log_transform(self, p, inp):
        df = inp["DataFrame"].copy()
        cols = [c.strip() for c in p.get("columns", "").split(",") if c.strip()]
        if not cols:
            cols = df.select_dtypes(include=np.number).columns.tolist()
        base = p.get("base", "natural (e)")
        for c in cols:
            if base == "log2":
                df[c] = np.log2(df[c] + 1)
            elif base == "log10":
                df[c] = np.log10(df[c] + 1)
            else:
                df[c] = np.log1p(df[c])
        return {"DataFrame": df}

    def _run_select_features(self, p, inp):
        df = inp["DataFrame"]
        target = p.get("target_col", "target")
        feat_str = p.get("feature_cols", "").strip()
        if feat_str:
            feat_cols = [c.strip() for c in feat_str.split(",") if c.strip()]
        else:
            feat_cols = [c for c in df.columns if c != target]
        X = df[feat_cols]
        y = df[target]
        return {"Features": X, "Target": y}

    def _run_train_test_split(self, p, inp):
        from sklearn.model_selection import train_test_split
        X, y = inp["Features"], inp["Target"]
        test_sz = float(p.get("test_size", 0.2))
        rs = int(p.get("random_state", 42))
        stratify = p.get("stratify", False)
        strat_arg = y if stratify else None
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_sz, random_state=rs, stratify=strat_arg)
        return {"X_train": X_tr, "X_test": X_te, "y_train": y_tr, "y_test": y_te}

    def _run_pca(self, p, inp):
        from sklearn.decomposition import PCA
        df = inp["DataFrame"]
        n = int(p.get("n_components", 2))
        num_cols = df.select_dtypes(include=np.number).columns
        pca = PCA(n_components=min(n, len(num_cols)))
        comps = pca.fit_transform(df[num_cols].fillna(0))
        out = pd.DataFrame(comps, columns=[f"PC{i+1}" for i in range(comps.shape[1])])
        return {"DataFrame": out}

    # ── Feature Extraction ────────────────────────────────────────
    def _run_tf_idf(self, p, inp):
        from sklearn.feature_extraction.text import TfidfVectorizer
        df = inp["DataFrame"]
        col = p.get("text_column", "text")
        mf = int(p.get("max_features", 1000))
        vec = TfidfVectorizer(max_features=mf, max_df=float(p.get("max_df", 0.95)),
                              min_df=int(p.get("min_df", 1)))
        mat = vec.fit_transform(df[col].fillna("").astype(str))
        out = pd.DataFrame(mat.toarray(), columns=[f"tfidf_{t}" for t in vec.get_feature_names_out()])
        return {"DataFrame": out}

    def _run_count_vectorizer(self, p, inp):
        from sklearn.feature_extraction.text import CountVectorizer
        df = inp["DataFrame"]
        col = p.get("text_column", "text")
        mf = int(p.get("max_features", 1000))
        vec = CountVectorizer(max_features=mf)
        mat = vec.fit_transform(df[col].fillna("").astype(str))
        out = pd.DataFrame(mat.toarray(), columns=[f"cv_{t}" for t in vec.get_feature_names_out()])
        return {"DataFrame": out}

    def _run_time_series_features(self, p, inp):
        df = inp["DataFrame"].copy()
        col = p.get("date_column", "date")
        df[col] = pd.to_datetime(df[col], errors="coerce")
        df["ts_year"] = df[col].dt.year
        df["ts_month"] = df[col].dt.month
        df["ts_day"] = df[col].dt.day
        df["ts_dayofweek"] = df[col].dt.dayofweek
        df["ts_hour"] = df[col].dt.hour
        df["ts_quarter"] = df[col].dt.quarter
        return {"DataFrame": df}

    # ── Models: Regression ────────────────────────────────────────
    def _fit_model(self, model, inp):
        X_tr, y_tr = inp.get("X_train"), inp.get("y_train")
        if X_tr is None or y_tr is None:
            raise ValueError("Need X_train and y_train inputs.")
        model.fit(X_tr, y_tr)
        return {"Model": model}

    def _run_linear_regression(self, p, inp):
        from sklearn.linear_model import LinearRegression
        return self._fit_model(LinearRegression(fit_intercept=p.get("fit_intercept", True)), inp)

    def _run_ridge_regression(self, p, inp):
        from sklearn.linear_model import Ridge
        return self._fit_model(Ridge(alpha=float(p.get("alpha", 1.0))), inp)

    def _run_lasso_regression(self, p, inp):
        from sklearn.linear_model import Lasso
        return self._fit_model(Lasso(alpha=float(p.get("alpha", 1.0))), inp)

    def _run_random_forest_regressor(self, p, inp):
        from sklearn.ensemble import RandomForestRegressor
        md = int(p.get("max_depth", 0)) or None
        return self._fit_model(RandomForestRegressor(n_estimators=int(p.get("n_estimators", 100)),
                                                     max_depth=md, random_state=int(p.get("random_state", 42))), inp)

    def _run_gradient_boosting_regressor(self, p, inp):
        from sklearn.ensemble import GradientBoostingRegressor
        return self._fit_model(GradientBoostingRegressor(
            n_estimators=int(p.get("n_estimators", 100)),
            learning_rate=float(p.get("learning_rate", 0.1)),
            max_depth=int(p.get("max_depth", 3))), inp)

    def _run_svr(self, p, inp):
        from sklearn.svm import SVR
        return self._fit_model(SVR(kernel=p.get("kernel", "rbf"),
                                   C=float(p.get("C", 1.0)),
                                   epsilon=float(p.get("epsilon", 0.1))), inp)

    # ── Models: Classification ────────────────────────────────────
    def _run_logistic_regression(self, p, inp):
        from sklearn.linear_model import LogisticRegression
        return self._fit_model(LogisticRegression(
            C=float(p.get("C", 1.0)), max_iter=int(p.get("max_iter", 1000)),
            solver=p.get("solver", "lbfgs")), inp)

    def _run_random_forest_classifier(self, p, inp):
        from sklearn.ensemble import RandomForestClassifier
        md = int(p.get("max_depth", 0)) or None
        return self._fit_model(RandomForestClassifier(
            n_estimators=int(p.get("n_estimators", 100)),
            max_depth=md, random_state=int(p.get("random_state", 42))), inp)

    def _run_svm_classifier(self, p, inp):
        from sklearn.svm import SVC
        return self._fit_model(SVC(kernel=p.get("kernel", "rbf"),
                                   C=float(p.get("C", 1.0)), probability=True), inp)

    def _run_knn_classifier(self, p, inp):
        from sklearn.neighbors import KNeighborsClassifier
        return self._fit_model(KNeighborsClassifier(
            n_neighbors=int(p.get("n_neighbors", 5)),
            weights=p.get("weights", "uniform")), inp)

    def _run_gradient_boosting_classifier(self, p, inp):
        from sklearn.ensemble import GradientBoostingClassifier
        return self._fit_model(GradientBoostingClassifier(
            n_estimators=int(p.get("n_estimators", 100)),
            learning_rate=float(p.get("learning_rate", 0.1)),
            max_depth=int(p.get("max_depth", 3))), inp)

    def _run_decision_tree_classifier(self, p, inp):
        from sklearn.tree import DecisionTreeClassifier
        md = int(p.get("max_depth", 0)) or None
        return self._fit_model(DecisionTreeClassifier(
            max_depth=md, criterion=p.get("criterion", "gini")), inp)

    def _run_naive_bayes(self, p, inp):
        from sklearn.naive_bayes import GaussianNB
        return self._fit_model(GaussianNB(), inp)
    
    # ── Deep Learning ─────────────────────────────────────────────
    def _run_mlp_classifier(self, p, inp):
        from sklearn.neural_network import MLPClassifier
        # Parse the comma-separated string into a tuple for the neural network layers
        layers_str = p.get("hidden_layers", "100,50")
        layers = tuple(int(x.strip()) for x in layers_str.split(",") if x.strip())
        
        model = MLPClassifier(
            hidden_layer_sizes=layers,
            max_iter=int(p.get("max_iter", 200)),
            learning_rate_init=float(p.get("learning_rate", 0.001)),
            random_state=42
        )
        return self._fit_model(model, inp)

    # ── Advanced Tuning & Ensembling ──────────────────────────────
    def _run_grid_search_cv(self, p, inp):
        from sklearn.model_selection import GridSearchCV
        from sklearn.base import clone
        import json
        
        base_model = inp["Model"]
        X, y = inp["Features"], inp["Target"]
        
        # Clone ensures we have a fresh, unfitted model for the grid search
        model_to_tune = clone(base_model)
        
        try:
            grid = json.loads(p.get("param_grid", "{}"))
        except json.JSONDecodeError:
            raise ValueError("Parameter grid must be valid JSON (e.g., {\"n_estimators\": [50, 100]})")
            
        cv = int(p.get("cv", 3))
        gs = GridSearchCV(model_to_tune, grid, cv=cv, scoring='accuracy')
        gs.fit(X, y)
        
        metrics = {
            "Best CV Score": float(gs.best_score_),
            "Best Parameters": gs.best_params_
        }
        
        # Returns the optimal model and the tuning metrics to the UI
        return {"Model": gs.best_estimator_, "Metrics": metrics}

    def _run_voting_classifier(self, p, inp):
        from sklearn.ensemble import VotingClassifier
        from sklearn.base import clone
        
        m1 = inp.get("Model 1")
        m2 = inp.get("Model 2")
        if m1 is None or m2 is None:
            raise ValueError("Voting Classifier requires both Model 1 and Model 2 connected.")
        
        # We clone the input models so the VotingClassifier can fit them together from scratch
        estimators = [
            ("model_1", clone(m1)),
            ("model_2", clone(m2))
        ]
        
        vc = VotingClassifier(estimators=estimators, voting=p.get("voting", "hard"))
        
        X_tr, y_tr = inp.get("X_train"), inp.get("y_train")
        if X_tr is None or y_tr is None:
            raise ValueError("Need X_train and y_train inputs to fit the Ensemble.")
            
        vc.fit(X_tr, y_tr)
        return {"Model": vc}

    # ── Models: Clustering / Anomaly ─────────────────────────────
    def _run_k_means(self, p, inp):
        from sklearn.cluster import KMeans
        df = inp["DataFrame"].copy()
        num_cols = df.select_dtypes(include=np.number).columns
        km = KMeans(n_clusters=int(p.get("n_clusters", 3)),
                    random_state=int(p.get("random_state", 42)),
                    n_init=int(p.get("n_init", 10)))
        df["cluster"] = km.fit_predict(df[num_cols].fillna(0))
        return {"DataFrame": df, "Model": km}

    def _run_dbscan(self, p, inp):
        from sklearn.cluster import DBSCAN
        df = inp["DataFrame"].copy()
        num_cols = df.select_dtypes(include=np.number).columns
        db = DBSCAN(eps=float(p.get("eps", 0.5)), min_samples=int(p.get("min_samples", 5)))
        df["cluster"] = db.fit_predict(df[num_cols].fillna(0))
        return {"DataFrame": df, "Model": db}

    def _run_isolation_forest(self, p, inp):
        from sklearn.ensemble import IsolationForest
        df = inp["DataFrame"].copy()
        num_cols = df.select_dtypes(include=np.number).columns
        iso = IsolationForest(contamination=float(p.get("contamination", 0.1)),
                              n_estimators=int(p.get("n_estimators", 100)))
        df["anomaly"] = iso.fit_predict(df[num_cols].fillna(0))
        df["anomaly_score"] = iso.score_samples(df[num_cols].fillna(0))
        return {"DataFrame": df, "Model": iso}

    # ── Evaluation ────────────────────────────────────────────────
    def _run_regression_metrics(self, p, inp):
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        model, X_te, y_te = inp["Model"], inp["X_test"], inp["y_test"]
        y_pred = model.predict(X_te)
        metrics = {
            "RMSE": float(np.sqrt(mean_squared_error(y_te, y_pred))),
            "MAE": float(mean_absolute_error(y_te, y_pred)),
            "R²": float(r2_score(y_te, y_pred)),
        }
        return {"Metrics": metrics}

    def _run_classification_metrics(self, p, inp):
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        model, X_te, y_te = inp["Model"], inp["X_test"], inp["y_test"]
        y_pred = model.predict(X_te)
        avg = "weighted"
        metrics = {
            "Accuracy": float(accuracy_score(y_te, y_pred)),
            "F1 (weighted)": float(f1_score(y_te, y_pred, average=avg, zero_division=0)),
            "Precision": float(precision_score(y_te, y_pred, average=avg, zero_division=0)),
            "Recall": float(recall_score(y_te, y_pred, average=avg, zero_division=0)),
        }
        try:
            from sklearn.metrics import roc_auc_score
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_te)
                classes = np.unique(y_te)
                if len(classes) == 2:
                    metrics["ROC AUC"] = float(roc_auc_score(y_te, proba[:, 1]))
                else:
                    metrics["ROC AUC (OvR)"] = float(roc_auc_score(y_te, proba, multi_class="ovr", average="weighted"))
        except Exception:
            pass
        return {"Metrics": metrics}

    def _run_cross_validation(self, p, inp):
        from sklearn.model_selection import cross_val_score
        model = inp["Model"]
        X, y = inp.get("Features"), inp.get("Target")
        if X is None or y is None:
            raise ValueError("Need Features and Target inputs")
        cv = int(p.get("cv", 5))
        scoring = p.get("scoring", "accuracy")
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        metrics = {
            f"CV Mean ({scoring})": float(scores.mean()),
            f"CV Std ({scoring})": float(scores.std()),
            "CV Scores": scores.tolist(),
        }
        return {"Metrics": metrics}

    def _run_feature_importance(self, p, inp):
        model = inp["Model"]
        X = inp.get("Features")
        if not hasattr(model, "feature_importances_"):
            raise ValueError("Model does not have feature_importances_ attribute")
        imp = model.feature_importances_
        cols = X.columns.tolist() if X is not None else [f"f{i}" for i in range(len(imp))]
        n = int(p.get("top_n", 15))
        idx = np.argsort(imp)[::-1][:n]
        metrics = {
            "feature_importance": {cols[i]: float(imp[i]) for i in idx}
        }
        return {"Metrics": metrics}

    # ── Prediction ────────────────────────────────────────────────
    def _run_predict(self, p, inp):
        model, df = inp["Model"], inp["DataFrame"].copy()
        num_cols = df.select_dtypes(include=np.number).columns
        preds = model.predict(df[num_cols].fillna(0))
        df[p.get("output_col", "prediction")] = preds
        return {"DataFrame": df}

    def _run_predict_proba(self, p, inp):
        model, df = inp["Model"], inp["DataFrame"].copy()
        num_cols = df.select_dtypes(include=np.number).columns
        prefix = p.get("output_prefix", "prob_")
        if not hasattr(model, "predict_proba"):
            raise ValueError("Model does not support predict_proba")
        proba = model.predict_proba(df[num_cols].fillna(0))
        for i in range(proba.shape[1]):
            df[f"{prefix}{i}"] = proba[:, i]
        return {"DataFrame": df}

    # ── Visualization ─────────────────────────────────────────────
    def _plot(self):
        fig = Figure(figsize=(8, 5), facecolor="#161B22")
        ax = fig.add_subplot(111)
        ax.set_facecolor("#161B22")
        for sp in ax.spines.values():
            sp.set_color("#30363D")
        ax.tick_params(colors="#8B949E")
        ax.xaxis.label.set_color("#8B949E")
        ax.yaxis.label.set_color("#8B949E")
        ax.title.set_color("#E6EDF3")
        return fig, ax

    def _run_histogram(self, p, inp):
        df = inp["DataFrame"]
        col = p.get("column", "").strip()
        bins = int(p.get("bins", 30))
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        if col and col in df.columns:
            cols_to_plot = [col]
        else:
            cols_to_plot = num_cols[:min(6, len(num_cols))]
        n = len(cols_to_plot)
        ncols_g = min(3, n)
        nrows_g = (n + ncols_g - 1) // ncols_g
        fig = Figure(figsize=(5 * ncols_g, 4 * nrows_g), facecolor="#161B22")
        colors = ["#58A6FF", "#3FB950", "#F78166", "#D2A8FF", "#FFA657", "#79C0FF"]
        for i, c in enumerate(cols_to_plot):
            ax = fig.add_subplot(nrows_g, ncols_g, i + 1)
            ax.set_facecolor("#0D1117")
            for sp in ax.spines.values(): sp.set_color("#30363D")
            ax.tick_params(colors="#8B949E", labelsize=8)
            ax.set_title(c, color="#E6EDF3", fontsize=12)
            ax.hist(df[c].dropna(), bins=bins, color=colors[i % len(colors)], alpha=0.85, edgecolor="#161B22")
        fig.tight_layout(pad=2)
        return {"_figure": fig}

    def _run_scatter_plot(self, p, inp):
        df = inp["DataFrame"]
        x_col = p.get("x_col", df.columns[0])
        y_col = p.get("y_col", df.columns[1] if len(df.columns) > 1 else df.columns[0])
        color_col = p.get("color_col", "").strip()
        fig, ax = self._plot()
        if color_col and color_col in df.columns:
            cats = df[color_col].unique()
            colors = plt.cm.tab10(np.linspace(0, 1, len(cats)))
            for cat, color in zip(cats, colors):
                mask = df[color_col] == cat
                ax.scatter(df.loc[mask, x_col], df.loc[mask, y_col], color=color, label=str(cat), alpha=0.7, s=25)
            ax.legend(facecolor="#21262D", labelcolor="#E6EDF3", fontsize=12)
        else:
            ax.scatter(df[x_col], df[y_col], color="#58A6FF", alpha=0.6, s=25)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f"{x_col} vs {y_col}")
        return {"_figure": fig}

    def _run_correlation_heatmap(self, p, inp):
        import matplotlib.colors as mcolors
        df = inp["DataFrame"]
        num_df = df.select_dtypes(include=np.number)
        corr = num_df.corr()
        n = len(corr)
        fig = Figure(figsize=(max(6, n * 0.6), max(5, n * 0.5)), facecolor="#161B22")
        ax = fig.add_subplot(111)
        ax.set_facecolor("#161B22")
        cmap = plt.cm.RdYlGn
        im = ax.imshow(corr, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=11, color="#8B949E")
        ax.set_yticklabels(corr.columns, fontsize=11, color="#8B949E")
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                        fontsize=10, color="white" if abs(corr.iloc[i, j]) > 0.5 else "#8B949E")
        fig.colorbar(im, ax=ax)
        ax.set_title("Correlation Heatmap", color="#E6EDF3")
        fig.tight_layout()
        return {"_figure": fig}

    def _run_confusion_matrix(self, p, inp):
        from sklearn.metrics import confusion_matrix
        model, X_te, y_te = inp["Model"], inp["X_test"], inp["y_test"]
        y_pred = model.predict(X_te)
        classes = np.unique(np.concatenate([y_te, y_pred]))
        cm = confusion_matrix(y_te, y_pred, labels=classes)
        n = len(classes)
        fig = Figure(figsize=(max(5, n * 1.1), max(4, n * 0.9)), facecolor="#161B22")
        ax = fig.add_subplot(111)
        ax.set_facecolor("#161B22")
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(classes, color="#8B949E"); ax.set_yticklabels(classes, color="#8B949E")
        for i in range(n):
            for j in range(n):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=10, color="white" if cm[i, j] > cm.max() / 2 else "#333")
        ax.set_xlabel("Predicted", color="#8B949E")
        ax.set_ylabel("Actual", color="#8B949E")
        ax.set_title("Confusion Matrix", color="#E6EDF3")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        return {"_figure": fig}

    def _run_roc_curve(self, p, inp):
        from sklearn.metrics import roc_curve, auc
        model, X_te, y_te = inp["Model"], inp["X_test"], inp["y_test"]
        if not hasattr(model, "predict_proba"):
            raise ValueError("Model must support predict_proba for ROC curve")
        fig, ax = self._plot()
        classes = np.unique(y_te)
        if len(classes) == 2:
            proba = model.predict_proba(X_te)[:, 1]
            fpr, tpr, _ = roc_curve(y_te, proba)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color="#58A6FF", lw=2, label=f"ROC AUC = {roc_auc:.3f}")
            ax.plot([0, 1], [0, 1], color="#484F58", lw=1, linestyle="--")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("ROC Curve")
            ax.legend(facecolor="#21262D", labelcolor="#E6EDF3")
        else:
            proba = model.predict_proba(X_te)
            colors = ["#58A6FF", "#3FB950", "#F78166", "#D2A8FF", "#FFA657"]
            for i, cls in enumerate(classes):
                y_bin = (y_te == cls).astype(int)
                fpr, tpr, _ = roc_curve(y_bin, proba[:, i])
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, color=colors[i % len(colors)], lw=2, label=f"Class {cls} AUC={roc_auc:.3f}")
            ax.plot([0, 1], [0, 1], color="#484F58", lw=1, linestyle="--")
            ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC Curve (OvR)")
            ax.legend(facecolor="#21262D", labelcolor="#E6EDF3", fontsize=10)
        return {"_figure": fig}

    def _run_residual_plot(self, p, inp):
        model, X_te, y_te = inp["Model"], inp["X_test"], inp["y_test"]
        y_pred = model.predict(X_te)
        resid = np.array(y_te) - y_pred
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor="#161B22")
        for ax in axes:
            ax.set_facecolor("#0D1117")
            for sp in ax.spines.values(): sp.set_color("#30363D")
            ax.tick_params(colors="#8B949E")
        axes[0].scatter(y_pred, resid, color="#58A6FF", alpha=0.5, s=20)
        axes[0].axhline(0, color="#F78166", lw=1.5, linestyle="--")
        axes[0].set_xlabel("Predicted", color="#8B949E"); axes[0].set_ylabel("Residuals", color="#8B949E")
        axes[0].set_title("Residuals vs Fitted", color="#E6EDF3")
        axes[1].hist(resid, bins=30, color="#3FB950", alpha=0.8, edgecolor="#161B22")
        axes[1].set_title("Residual Distribution", color="#E6EDF3")
        axes[1].set_xlabel("Residual", color="#8B949E")
        fig.tight_layout()
        # Convert matplotlib figure to our Figure type
        canvas = FigureCanvas(fig)
        return {"_figure": fig}

    def _run_box_plot(self, p, inp):
        df = inp["DataFrame"]
        cols_str = p.get("columns", "").strip()
        cols = [c.strip() for c in cols_str.split(",") if c.strip()] if cols_str else \
               df.select_dtypes(include=np.number).columns.tolist()[:8]
        fig = Figure(figsize=(max(6, len(cols) * 1.2), 5), facecolor="#161B22")
        ax = fig.add_subplot(111)
        ax.set_facecolor("#0D1117")
        for sp in ax.spines.values(): sp.set_color("#30363D")
        ax.tick_params(colors="#8B949E")
        data = [df[c].dropna().values for c in cols]
        bp = ax.boxplot(data, patch_artist=True, notch=False,
                        medianprops=dict(color="#F78166", lw=2),
                        whiskerprops=dict(color="#8B949E"),
                        capprops=dict(color="#8B949E"),
                        flierprops=dict(marker='o', color="#FFA657", alpha=0.3, markersize=3))
        colors = ["#58A6FF", "#3FB950", "#D2A8FF", "#FFA657", "#F78166"]
        for patch, color in zip(bp['boxes'], colors * (len(cols) // len(colors) + 1)):
            patch.set_facecolor(color); patch.set_alpha(0.7)
        ax.set_xticks(range(1, len(cols) + 1))
        ax.set_xticklabels(cols, rotation=30, ha="right", fontsize=11, color="#8B949E")
        ax.set_title("Box Plot", color="#E6EDF3")
        fig.tight_layout()
        return {"_figure": fig}

    def _run_show_metrics(self, p, inp):
        metrics = inp.get("Metrics", {})
        return {"_metrics": metrics}

    def _run_data_summary(self, p, inp):
        df = inp["DataFrame"]
        return {"_metrics": {
            "Shape": f"{df.shape[0]} rows × {df.shape[1]} columns",
            "Columns": list(df.columns),
            "Dtypes": {str(k): str(v) for k, v in df.dtypes.items()},
            "Missing Values": df.isnull().sum().to_dict(),
            "Numeric Summary": df.describe().to_dict(),
        }}


# ─────────────────────────────────────────────────────────────────
#  DATA MODELS
# ─────────────────────────────────────────────────────────────────

class Port:
    def __init__(self, name: str, is_input: bool, node: 'Node', index: int):
        self.name = name
        self.is_input = is_input
        self.node = node
        self.index = index
        self.connections: List['Wire'] = []

    @property
    def scene_pos(self) -> QPointF:
        r = self.node.rect
        x = r.left() - 2 if self.is_input else r.right() + 2
        total = len(self.node.input_ports if self.is_input else self.node.output_ports)
        spacing = (r.height() - HEADER_H) / (total + 1)
        y = r.top() + HEADER_H + spacing * (self.index + 1)
        return QPointF(x, y)


class Wire:
    def __init__(self, src: Port, dst: Port):
        self.id = str(uuid.uuid4())
        self.src = src
        self.dst = dst
        src.connections.append(self)
        dst.connections.append(self)

    def remove(self):
        if self in self.src.connections:
            self.src.connections.remove(self)
        if self in self.dst.connections:
            self.dst.connections.remove(self)


class Node:
    def __init__(self, block_name: str, x: float = 100, y: float = 100):
        self.id = str(uuid.uuid4())
        self.block_name = block_name
        self.block_def = BLOCK_DEFS[block_name]
        self.params: Dict[str, Any] = {
            k: v.get("default", v.get("options", [""])[0] if v["type"] == "combo" else "")
            for k, v in self.block_def.get("params", {}).items()
        }
        w = max(MIN_NODE_W, len(block_name) * 9 + 60)
        self.rect = QRectF(x, y, w, NODE_H + max(
            len(self.block_def["inputs"]),
            len(self.block_def["outputs"]), 1) * 18)
        self.input_ports = [Port(n, True, self, i) for i, n in enumerate(self.block_def["inputs"])]
        self.output_ports = [Port(n, False, self, i) for i, n in enumerate(self.block_def["outputs"])]
        self.state = "idle"  # idle | running | ok | error
        self.error_msg = ""
        self.result: Dict[str, Any] = {}
        self.category = self.block_def["category"]

    def move(self, dx: float, dy: float):
        self.rect.moveLeft(self.rect.left() + dx)
        self.rect.moveTop(self.rect.top() + dy)

    def set_pos(self, x: float, y: float):
        self.rect.moveLeft(x)
        self.rect.moveTop(y)

    @property
    def icon(self) -> str:
        return self.block_def.get("icon", "⬡")


# ─────────────────────────────────────────────────────────────────
#  CANVAS
# ─────────────────────────────────────────────────────────────────

class Canvas(QWidget):
    node_selected = pyqtSignal(object)
    run_requested = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.nodes: List[Node] = []
        self.wires: List[Wire] = []
        self.selected_nodes: List[Node] = []
        self.scale = 1.0
        self.offset = QPointF(0, 0)
        self.drag_node: Optional[Node] = None
        self.drag_start: Optional[QPointF] = None
        self.pan_start: Optional[QPoint] = None
        self.pan_offset_start: Optional[QPointF] = None
        self.wire_start: Optional[Port] = None
        self.wire_mouse: Optional[QPointF] = None
        self.rubber_start: Optional[QPointF] = None
        self.rubber_rect: Optional[QRectF] = None
        self.setAcceptDrops(True)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(800, 600)

    # ── coordinate helpers ────────────────────────────────────────
    def to_world(self, screen: QPointF) -> QPointF:
        return (screen - self.offset) / self.scale

    def to_screen(self, world: QPointF) -> QPointF:
        return world * self.scale + self.offset

    def port_at(self, world: QPointF) -> Optional[Port]:
        for node in self.nodes:
            for port in node.input_ports + node.output_ports:
                sp = port.scene_pos
                if (sp - world).manhattanLength() < PORT_HIT:
                    return port
        return None

    def node_at(self, world: QPointF) -> Optional[Node]:
        for node in reversed(self.nodes):
            if node.rect.contains(world):
                return node
        return None

    # ── drag & drop ───────────────────────────────────────────────
    def dragEnterEvent(self, e: QDragEnterEvent):
        if e.mimeData().hasText():
            e.accept()

    def dropEvent(self, e: QDropEvent):
        name = e.mimeData().text()
        if name in BLOCK_DEFS:
            world = self.to_world(QPointF(e.pos()))
            snapped_x = round(world.x() / GRID_SIZE) * GRID_SIZE
            snapped_y = round(world.y() / GRID_SIZE) * GRID_SIZE
            node = Node(name, snapped_x, snapped_y)
            self.nodes.append(node)
            self.selected_nodes = [node]
            self.node_selected.emit(node)
            self.update()
        e.accept()

    # ── mouse ─────────────────────────────────────────────────────
    def mousePressEvent(self, e: QMouseEvent):
        world = self.to_world(QPointF(e.pos()))
        if e.button() == Qt.LeftButton:
            port = self.port_at(world)
            if port:
                # disconnect existing if input already connected
                if port.is_input and port.connections:
                    old_wire = port.connections[0]
                    old_wire.remove()
                    self.wires.remove(old_wire)
                self.wire_start = port
                self.wire_mouse = world
                self.update()
                return
            node = self.node_at(world)
            if node:
                if e.modifiers() & Qt.ControlModifier:
                    if node in self.selected_nodes:
                        self.selected_nodes.remove(node)
                    else:
                        self.selected_nodes.append(node)
                else:
                    if node not in self.selected_nodes:
                        self.selected_nodes = [node]
                self.drag_node = node
                self.drag_start = world
                self.node_selected.emit(node if len(self.selected_nodes) == 1 else None)
            else:
                self.selected_nodes = []
                self.node_selected.emit(None)
                self.rubber_start = world
                self.rubber_rect = QRectF(world, QSizeF(0, 0))
            self.update()
        elif e.button() == Qt.MiddleButton or (e.button() == Qt.LeftButton and e.modifiers() & Qt.AltModifier):
            self.pan_start = e.pos()
            self.pan_offset_start = QPointF(self.offset)
        elif e.button() == Qt.RightButton:
            self._show_context_menu(e.pos(), world)

    def mouseMoveEvent(self, e: QMouseEvent):
        world = self.to_world(QPointF(e.pos()))
        if self.wire_start:
            self.wire_mouse = world
            self.update()
        elif self.drag_node and self.drag_start:
            delta = world - self.drag_start
            for node in self.selected_nodes:
                node.move(delta.x(), delta.y())
            self.drag_start = world
            self.update()
        elif self.pan_start:
            delta = e.pos() - self.pan_start
            self.offset = self.pan_offset_start + QPointF(delta)
            self.update()
        elif self.rubber_start:
            self.rubber_rect = QRectF(self.rubber_start, world).normalized()
            self.update()

    def mouseReleaseEvent(self, e: QMouseEvent):
        if e.button() == Qt.LeftButton:
            if self.wire_start:
                world = self.to_world(QPointF(e.pos()))
                end_port = self.port_at(world)
                if end_port and end_port != self.wire_start:
                    src, dst = self.wire_start, end_port
                    if src.is_input and not dst.is_input:
                        src, dst = dst, src
                    if not src.is_input and dst.is_input and src.node != dst.node:
                        # remove existing connections to dst
                        for w in list(dst.connections):
                            w.remove()
                            self.wires.remove(w)
                        wire = Wire(src, dst)
                        self.wires.append(wire)
                self.wire_start = None
                self.wire_mouse = None
                self.update()
            elif self.rubber_rect:
                self.selected_nodes = [n for n in self.nodes if self.rubber_rect.intersects(n.rect)]
                if len(self.selected_nodes) == 1:
                    self.node_selected.emit(self.selected_nodes[0])
                else:
                    self.node_selected.emit(None)
                self.rubber_rect = None
                self.rubber_start = None
                self.update()
            self.drag_node = None
            self.drag_start = None
            self.pan_start = None

    def wheelEvent(self, e: QWheelEvent):
        factor = 1.12 if e.angleDelta().y() > 0 else 1 / 1.12
        screen_pt = QPointF(e.pos())
        world_pt = self.to_world(screen_pt)
        self.scale = max(0.2, min(3.0, self.scale * factor))
        self.offset = screen_pt - world_pt * self.scale
        self.update()

    def keyPressEvent(self, e: QKeyEvent):
        if e.key() == Qt.Key_Delete or e.key() == Qt.Key_Backspace:
            self._delete_selected()
        elif e.key() == Qt.Key_A and e.modifiers() & Qt.ControlModifier:
            self.selected_nodes = list(self.nodes)
            self.update()
        elif e.key() == Qt.Key_F:
            self.fit_view()

    def _delete_selected(self):
        for node in list(self.selected_nodes):
            for port in node.input_ports + node.output_ports:
                for wire in list(port.connections):
                    wire.remove()
                    if wire in self.wires:
                        self.wires.remove(wire)
            self.nodes.remove(node)
        self.selected_nodes = []
        self.node_selected.emit(None)
        self.update()

    def _show_context_menu(self, screen_pos: QPoint, world: QPointF):
        node = self.node_at(world)
        menu = QMenu(self)
        menu.setStyleSheet(f"""
            QMenu {{ background:{THEME['bg_light']}; color:{THEME['text']}; border:1px solid {THEME['border']}; border-radius:6px; padding:4px; }}
            QMenu::item {{ padding:6px 20px; border-radius:4px; }}
            QMenu::item:selected {{ background:{THEME['accent']}; color:white; }}
        """)
        if node:
            if len(self.selected_nodes) <= 1:
                self.selected_nodes = [node]
            a = menu.addAction(f"🗑️  Delete {'Selected' if len(self.selected_nodes) > 1 else node.block_name}")
            a.triggered.connect(self._delete_selected)
            menu.addSeparator()
            a2 = menu.addAction("▶  Run From Here")
            a2.triggered.connect(self.run_requested.emit)
        else:
            a = menu.addAction("🗑️  Clear All")
            a.triggered.connect(self.clear_all)
            a2 = menu.addAction("📐  Fit View")
            a2.triggered.connect(self.fit_view)
        menu.exec_(self.mapToGlobal(screen_pos))

    def clear_all(self):
        self.nodes.clear()
        self.wires.clear()
        self.selected_nodes.clear()
        self.node_selected.emit(None)
        self.update()

    def fit_view(self):
        if not self.nodes:
            return
        min_x = min(n.rect.left() for n in self.nodes)
        min_y = min(n.rect.top() for n in self.nodes)
        max_x = max(n.rect.right() for n in self.nodes)
        max_y = max(n.rect.bottom() for n in self.nodes)
        pad = 60
        world_w = max_x - min_x + pad * 2
        world_h = max_y - min_y + pad * 2
        sx = self.width() / world_w
        sy = self.height() / world_h
        self.scale = max(0.2, min(2.0, min(sx, sy)))
        self.offset = QPointF(
            -((min_x - pad) * self.scale),
            -((min_y - pad) * self.scale)
        )
        self.update()

    # ── painting ─────────────────────────────────────────────────
    def paintEvent(self, e):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.TextAntialiasing)

        # background
        painter.fillRect(self.rect(), QColor(THEME["bg_dark"]))

        # grid
        painter.save()
        painter.translate(self.offset)
        painter.scale(self.scale, self.scale)
        self._draw_grid(painter)
        self._draw_wires(painter)
        self._draw_nodes(painter)
        if self.wire_start and self.wire_mouse:
            self._draw_temp_wire(painter)
        if self.rubber_rect:
            self._draw_rubber(painter)
        painter.restore()

    def _draw_grid(self, p: QPainter):
        vp = QRectF(self.rect())
        tl = self.to_world(vp.topLeft())
        br = self.to_world(vp.bottomRight())
        p.setPen(QPen(QColor(THEME["grid"]), 0.5 / self.scale))
        xs = int(tl.x() // GRID_SIZE) * GRID_SIZE
        while xs < br.x():
            p.drawLine(QLineF(xs, tl.y(), xs, br.y()))
            xs += GRID_SIZE
        ys = int(tl.y() // GRID_SIZE) * GRID_SIZE
        while ys < br.y():
            p.drawLine(QLineF(tl.x(), ys, br.x(), ys))
            ys += GRID_SIZE

    def _draw_wires(self, p: QPainter):
        for wire in self.wires:
            sp = wire.src.scene_pos
            ep = wire.dst.scene_pos
            self._draw_bezier(p, sp, ep, QColor(THEME["wire"]), 2)

    def _draw_bezier(self, p: QPainter, sp: QPointF, ep: QPointF, color: QColor, width: float):
        cx = abs(ep.x() - sp.x()) * 0.5
        path = QPainterPath(sp)
        path.cubicTo(QPointF(sp.x() + cx, sp.y()),
                     QPointF(ep.x() - cx, ep.y()),
                     ep)
        pen = QPen(color, width / self.scale)
        pen.setCapStyle(Qt.RoundCap)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        p.drawPath(path)
        # arrowhead
        p.setBrush(QBrush(color))
        p.setPen(Qt.NoPen)
        arrow_size = 6 / self.scale
        dx = ep.x() - (ep.x() - cx * 0.3)
        dy = ep.y() - (ep.y() - 0.0)
        angle = 0  # we'll skip angle calc for simplicity

    def _draw_temp_wire(self, p: QPainter):
        sp = self.wire_start.scene_pos
        ep = self.wire_mouse
        self._draw_bezier(p, sp, ep, QColor(THEME["wire_active"]), 1.5)

    def _draw_rubber(self, p: QPainter):
        pen = QPen(QColor(THEME["selection"]), 1 / self.scale)
        pen.setStyle(Qt.DashLine)
        p.setPen(pen)
        p.setBrush(QBrush(QColor(THEME["selection"] + "22")))
        p.drawRect(self.rubber_rect)

    def _draw_nodes(self, p: QPainter):
        for node in self.nodes:
            selected = node in self.selected_nodes
            self._draw_node(p, node, selected)

    def _draw_node(self, p: QPainter, node: Node, selected: bool):
        r = node.rect
        cat_color = QColor(BLOCK_CATEGORIES.get(node.category, "#555"))

        # shadow
        shadow_rect = r.adjusted(3 / self.scale, 4 / self.scale, 3 / self.scale, 4 / self.scale)
        p.setBrush(QBrush(QColor(0, 0, 0, 60)))
        p.setPen(Qt.NoPen)
        p.drawRoundedRect(shadow_rect, 8, 8)

        # body gradient
        grad = QLinearGradient(r.topLeft(), r.bottomLeft())
        grad.setColorAt(0, QColor(THEME["bg_light"]))
        grad.setColorAt(1, QColor(THEME["bg_mid"]))
        p.setBrush(QBrush(grad))

        if selected:
            p.setPen(QPen(QColor(THEME["selection"]), 2 / self.scale))
        else:
            p.setPen(QPen(QColor(THEME["border"]), 1 / self.scale))
        p.drawRoundedRect(r, 8, 8)

        # header
        header_rect = QRectF(r.left(), r.top(), r.width(), HEADER_H)
        header_path = QPainterPath()
        header_path.addRoundedRect(header_rect, 8, 8)
        clip_rect = QRectF(r.left(), r.top() + HEADER_H / 2, r.width(), HEADER_H / 2)
        header_path.addRect(clip_rect)
        p.setBrush(QBrush(cat_color.darker(140)))
        p.setPen(Qt.NoPen)
        p.drawPath(header_path)

        # category strip
        strip = QRectF(r.left(), r.top(), 4, r.height())
        strip_path = QPainterPath()
        strip_path.addRoundedRect(strip, 4, 4)
        clip2 = QRectF(r.left() + 2, r.top(), 4, r.height())
        strip_path.addRect(clip2)
        p.setBrush(QBrush(cat_color))
        p.drawPath(strip_path)

        # state indicator
        state_colors = {"idle": "#484F58", "running": "#FFA657", "ok": "#3FB950", "error": "#F78166"}
        dot_color = QColor(state_colors.get(node.state, "#484F58"))
        p.setBrush(QBrush(dot_color))
        p.setPen(Qt.NoPen)
        dot_r = 4 / self.scale
        p.drawEllipse(QPointF(r.right() - 10, r.top() + HEADER_H / 2), dot_r, dot_r)

        # icon + title
        font = QFont("Segoe UI", max(7, int(9 / self.scale)))
        font.setBold(True)
        p.setFont(font)
        p.setPen(QPen(QColor(THEME["text"])))
        title_rect = QRectF(r.left() + 10, r.top(), r.width() - 24, HEADER_H)
        p.drawText(title_rect, Qt.AlignVCenter | Qt.AlignLeft,
                   f"{node.icon}  {node.block_name}")

        # ports
        font2 = QFont("Segoe UI", max(6, int(7.5 / self.scale)))
        p.setFont(font2)
        for port in node.input_ports:
            self._draw_port(p, port, cat_color)
        for port in node.output_ports:
            self._draw_port(p, port, cat_color)

    def _draw_port(self, p: QPainter, port: Port, cat_color: QColor):
        sp = port.scene_pos
        connected = bool(port.connections)
        r = PORT_RADIUS / self.scale

        if connected:
            p.setBrush(QBrush(cat_color))
        else:
            p.setBrush(QBrush(QColor(THEME["bg_dark"])))

        p.setPen(QPen(cat_color, 1.5 / self.scale))
        p.drawEllipse(sp, r, r)

        # port label
        font = QFont("Segoe UI", max(5, int(6.5 / self.scale)))
        p.setFont(font)
        p.setPen(QPen(QColor(THEME["text_dim"])))
        label_x = sp.x() + (r + 3 / self.scale) if not port.is_input else sp.x() - r - 3 / self.scale
        align = Qt.AlignLeft | Qt.AlignVCenter if not port.is_input else Qt.AlignRight | Qt.AlignVCenter
        label_rect = QRectF(label_x - (50 if port.is_input else 0), sp.y() - 8,
                            52, 16)
        p.drawText(label_rect, align, port.name)


# ─────────────────────────────────────────────────────────────────
#  PANEL WIDGETS
# ─────────────────────────────────────────────────────────────────

PANEL_STYLE = f"""
    QWidget {{ background: {THEME['bg_panel']}; color: {THEME['text']}; font-family: 'Segoe UI', sans-serif; }}
    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
        background: {THEME['bg_light']}; border: 1px solid {THEME['border']};
        border-radius: 4px; padding: 4px 8px; color: {THEME['text']};
    }}
    QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
        border-color: {THEME['accent']};
    }}
    QLabel {{ color: {THEME['text_dim']}; font-size: 12px; }}
    QPushButton {{
        background: {THEME['bg_light']}; border: 1px solid {THEME['border']};
        border-radius: 5px; padding: 5px 12px; color: {THEME['text']};
    }}
    QPushButton:hover {{ background: {THEME['accent']}; color: white; border-color: {THEME['accent']}; }}
    QScrollArea {{ border: none; }}
    QScrollBar:vertical {{ background: {THEME['bg_dark']}; width: 6px; }}
    QScrollBar::handle:vertical {{ background: {THEME['border']}; border-radius: 3px; }}
"""


class ParamEditor(QWidget):
    params_changed = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.setStyleSheet(PANEL_STYLE)
        self._node: Optional[Node] = None
        self._widgets: Dict[str, QWidget] = {}
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(12, 12, 12, 12)
        self._layout.setSpacing(8)
        self._placeholder = QLabel("Select a node to edit its parameters")
        self._placeholder.setAlignment(Qt.AlignCenter)
        self._placeholder.setStyleSheet(f"color: {THEME['text_muted']}; font-size: 16px;")
        self._layout.addWidget(self._placeholder)
        self._layout.addStretch()

    def set_node(self, node: Optional[Node]):
        self._node = node
        self._widgets.clear()
        # clear layout
        while self._layout.count():
            item = self._layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if node is None:
            self._placeholder = QLabel("Select a node to edit its parameters")
            self._placeholder.setAlignment(Qt.AlignCenter)
            self._placeholder.setStyleSheet(f"color: {THEME['text_muted']}; font-size: 16px;")
            self._layout.addWidget(self._placeholder)
            self._layout.addStretch()
            return

        # Header
        cat_color = BLOCK_CATEGORIES.get(node.category, "#555")
        header = QLabel(f"{node.icon}  {node.block_name}")
        header.setStyleSheet(f"color: {THEME['text']}; font-size: 16px; font-weight: bold; "
                             f"border-bottom: 2px solid {cat_color}; padding-bottom: 6px; margin-bottom: 6px;")
        self._layout.addWidget(header)

        desc = QLabel(node.block_def.get("description", ""))
        desc.setWordWrap(True)
        desc.setStyleSheet(f"color: {THEME['text']}; font-size: 15px; margin-bottom: 8px;")
        self._layout.addWidget(desc)

        params = node.block_def.get("params", {})
        if not params:
            empty = QLabel("No parameters")
            empty.setStyleSheet(f"color: {THEME['text_muted']}; font-style: italic; font-size: 16px;")
            self._layout.addWidget(empty)
        else:
            for key, meta in params.items():
                row = QWidget()
                row.setStyleSheet("QWidget { background: transparent; }")
                rl = QVBoxLayout(row)
                rl.setContentsMargins(0, 0, 0, 0)
                rl.setSpacing(3)
                label = QLabel(meta.get("label", key))
                label.setStyleSheet(f"color: {THEME['text']}; font-size: 15px;")
                rl.addWidget(label)
                current = node.params.get(key, meta.get("default", ""))
                widget = self._make_widget(key, meta, current)
                rl.addWidget(widget)
                self._layout.addWidget(row)
                self._widgets[key] = widget

        self._layout.addStretch()

    def _make_widget(self, key: str, meta: dict, current) -> QWidget:
        t = meta.get("type", "str")
        if t == "combo":
            w = QComboBox()
            w.addItems(meta.get("options", []))
            if str(current) in meta.get("options", []):
                w.setCurrentText(str(current))
            w.currentTextChanged.connect(lambda v, k=key: self._update(k, v))
            return w
        elif t == "bool":
            w = QCheckBox()
            w.setChecked(bool(current))
            w.setStyleSheet(f"color: {THEME['text']};")
            w.stateChanged.connect(lambda s, k=key: self._update(k, s == Qt.Checked))
            return w
        elif t == "int":
            w = QSpinBox()
            w.setRange(meta.get("min", 0), meta.get("max", 100000))
            w.setValue(int(current) if current != "" else 0)
            w.valueChanged.connect(lambda v, k=key: self._update(k, v))
            return w
        elif t == "float":
            w = QDoubleSpinBox()
            w.setRange(meta.get("min", -1e9), meta.get("max", 1e9))
            w.setSingleStep(0.01)
            w.setDecimals(4)
            w.setValue(float(current) if current != "" else 0.0)
            w.valueChanged.connect(lambda v, k=key: self._update(k, v))
            return w
        elif t in ("file", "save_file"):
            container = QWidget()
            container.setStyleSheet("background: transparent;")
            hl = QHBoxLayout(container)
            hl.setContentsMargins(0, 0, 0, 0)
            le = QLineEdit(str(current))
            le.setPlaceholderText("Path to file...")
            le.textChanged.connect(lambda v, k=key: self._update(k, v))
            btn = QPushButton("Browse")
            btn.setFixedWidth(65)
            filt = meta.get("filter", "All Files (*)")
            is_save = t == "save_file"
            btn.clicked.connect(lambda _, le=le, f=filt, s=is_save: self._browse(le, f, s))
            hl.addWidget(le)
            hl.addWidget(btn)
            return container
        else:
            w = QLineEdit(str(current))
            w.textChanged.connect(lambda v, k=key: self._update(k, v))
            return w

    def _browse(self, le: QLineEdit, filt: str, save: bool):
        if save:
            path, _ = QFileDialog.getSaveFileName(self, "Save File", "", filt)
        else:
            path, _ = QFileDialog.getOpenFileName(self, "Open File", "", filt)
        if path:
            le.setText(path)

    def _update(self, key: str, value):
        if self._node:
            self._node.params[key] = value
            self.params_changed.emit(self._node.params)


# ─────────────────────────────────────────────────────────────────
#  BLOCK LIBRARY PANEL
# ─────────────────────────────────────────────────────────────────

class BlockLibrary(QWidget):
    def __init__(self):
        super().__init__()
        self.setStyleSheet(PANEL_STYLE)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        title = QLabel("Block Library")
        title.setStyleSheet(f"color: {THEME['text']}; font-size: 16px; font-weight: bold; "
                            f"padding-bottom: 4px; border-bottom: 1px solid {THEME['border']};")
        layout.addWidget(title)

        search = QLineEdit()
        search.setPlaceholderText("🔍 Search blocks...")
        search.textChanged.connect(self._filter)
        layout.addWidget(search)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._content = QWidget()
        self._content.setStyleSheet("background: transparent;")
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setSpacing(2)
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        scroll.setWidget(self._content)
        layout.addWidget(scroll)

        self._all_items: List[Tuple[str, QWidget]] = []
        self._build_library()

    def _build_library(self):
        # Group by category
        groups: Dict[str, List[str]] = {}
        for name, defn in BLOCK_DEFS.items():
            cat = defn["category"]
            groups.setdefault(cat, []).append(name)

        for cat, blocks in groups.items():
            cat_color = BLOCK_CATEGORIES.get(cat, "#555")
            # Category header
            cat_btn = QLabel(cat)
            cat_btn.setStyleSheet(f"""
                color: {cat_color}; font-size: 16px; font-weight: bold;
                background: transparent; padding: 10px 4px 6px 4px;
            """)
            self._content_layout.addWidget(cat_btn)

            for block_name in sorted(blocks):
                item = self._make_item(block_name, BLOCK_DEFS[block_name], cat_color)
                self._content_layout.addWidget(item)
                self._all_items.append((block_name.lower(), item))

        self._content_layout.addStretch()

    def _make_item(self, name: str, defn: dict, cat_color: str) -> QWidget:
        w = QLabel(f"  {defn.get('icon', '⬡')}  {name}")
        w.setFixedHeight(36)
        w.setStyleSheet(f"""
            QLabel {{
                background: {THEME['bg_light']}; color: {THEME['text']};
                border-radius: 5px; border-left: 3px solid {cat_color};
                font-size: 16px; padding: 2px 4px;
            }}
            QLabel:hover {{ background: {THEME['bg_dark']}; }}
        """)
        w.setToolTip(defn.get("description", ""))
        w.setCursor(Qt.OpenHandCursor)
        w.setProperty("block_name", name)

        # Make draggable
        def make_drag(widget, bname):
            def press(e):
                if e.button() == Qt.LeftButton:
                    drag = QDrag(widget)
                    mime = QMimeData()
                    mime.setText(bname)
                    drag.setMimeData(mime)
                    pix = QPixmap(widget.size())
                    pix.fill(Qt.transparent)
                    widget.render(pix)
                    drag.setPixmap(pix)
                    drag.setHotSpot(e.pos())
                    drag.exec_(Qt.CopyAction)
            return press
        w.mousePressEvent = make_drag(w, name)
        return w

    def _filter(self, text: str):
        t = text.lower()
        for bname, item in self._all_items:
            item.setVisible(t in bname or not t)


# ─────────────────────────────────────────────────────────────────
#  OUTPUT / RESULT VIEWER
# ─────────────────────────────────────────────────────────────────

class ResultViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setStyleSheet(PANEL_STYLE)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        header = QWidget()
        header.setStyleSheet(f"background: {THEME['bg_mid']}; border-bottom: 1px solid {THEME['border']};")
        hl = QHBoxLayout(header)
        hl.setContentsMargins(12, 8, 12, 8)
        hl_title = QLabel("▶  Output Console")
        hl_title.setStyleSheet(f"color: {THEME['text']}; font-weight: bold; font-size: 18px;")
        hl.addWidget(hl_title)
        hl.addStretch()
        self._clear_btn = QPushButton("Clear")
        self._clear_btn.clicked.connect(self.clear)
        hl.addWidget(self._clear_btn)
        layout.addWidget(header)

        self._tabs = QTabWidget()
        self._tabs.setStyleSheet(f"""
            QTabWidget::pane {{ border: none; background: {THEME['bg_dark']}; }}
            QTabBar::tab {{ background: {THEME['bg_mid']}; color: {THEME['text_dim']}; font-size: 16px;
                           padding: 8px 16px; border: none; border-right: 1px solid {THEME['border']}; }}
            QTabBar::tab:selected {{ background: {THEME['bg_dark']}; color: {THEME['text']}; border-top: 2px solid {THEME['accent']}; }}
        """)
        layout.addWidget(self._tabs)

        # Log tab
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setStyleSheet(f"background: {THEME['bg_dark']}; color: {THEME['text']}; "
                                f"font-family: 'Consolas', monospace; font-size: 16px; border: none;")
        self._tabs.addTab(self._log, "📋 Log")

        # Plot area
        self._plot_area = QScrollArea()
        self._plot_area.setWidgetResizable(True)
        self._plot_widget = QWidget()
        self._plot_widget.setStyleSheet(f"background: {THEME['bg_dark']};")
        self._plot_layout = QVBoxLayout(self._plot_widget)
        self._plot_area.setWidget(self._plot_widget)
        self._tabs.addTab(self._plot_area, "📊 Plots")

        # Data preview
        self._data_table = QTableWidget()
        self._data_table.setStyleSheet(f"""
            QTableWidget {{ background: {THEME['bg_dark']}; color: {THEME['text']}; gridline-color: {THEME['border']}; border: none; font-size: 16px;}}
            QHeaderView::section {{ background: {THEME['bg_mid']}; color: {THEME['text_dim']}; padding: 4px; border: none; border-bottom: 1px solid {THEME['border']}; font-size: 16px; font-weight: bold; }}
        """)
        self._tabs.addTab(self._data_table, "📋 Data Preview")

    def log(self, msg: str, level="info"):
        colors = {"info": THEME["text"], "ok": "#3FB950", "error": "#F78166", "warn": "#FFA657"}
        color = colors.get(level, THEME["text"])
        self._log.append(f'<span style="color:{color};">{msg}</span>')
        self._log.moveCursor(self._log.textCursor().End)
        self._tabs.setCurrentIndex(0)

    def show_figure(self, fig, title="Plot"):
        canvas = FigureCanvas(fig)
        canvas.setStyleSheet(f"background: {THEME['bg_dark']};")
        canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        canvas.setFixedHeight(380)

        container = QWidget()
        container.setStyleSheet(f"background: {THEME['bg_mid']}; border-radius: 6px; margin: 4px;")
        cl = QVBoxLayout(container)
        cl.setContentsMargins(4, 4, 4, 4)
        t = QLabel(title)
        t.setStyleSheet(f"color: {THEME['text_dim']}; font-size: 14px; padding: 2px 6px;")
        cl.addWidget(t)
        cl.addWidget(canvas)
        self._plot_layout.addWidget(container)
        canvas.draw()
        self._tabs.setCurrentIndex(1)

    def show_dataframe(self, df: pd.DataFrame, label=""):
        self._data_table.clear()
        preview = df.head(500)
        self._data_table.setRowCount(len(preview))
        self._data_table.setColumnCount(len(preview.columns))
        self._data_table.setHorizontalHeaderLabels([str(c) for c in preview.columns])
        for i, row in enumerate(preview.itertuples(index=False)):
            for j, val in enumerate(row):
                item = QTableWidgetItem(str(val))
                item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
                self._data_table.setItem(i, j, item)
        self._data_table.resizeColumnsToContents()
        self._tabs.setCurrentIndex(2)

    def show_metrics(self, metrics: dict, title="Metrics"):
        self.log(f"── {title} ──", "info")
        for k, v in metrics.items():
            if isinstance(v, float):
                self.log(f"  {k}: <b>{v:.4f}</b>", "ok")
            elif isinstance(v, dict):
                self.log(f"  {k}:", "info")
                for kk, vv in v.items():
                    if isinstance(vv, float):
                        self.log(f"    {kk}: {vv:.4f}", "ok")
                    else:
                        self.log(f"    {kk}: {vv}", "info")
            else:
                self.log(f"  {k}: {v}", "info")

    def clear(self):
        self._log.clear()
        while self._plot_layout.count():
            item = self._plot_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._data_table.clear()
        self._data_table.setRowCount(0)
        self._data_table.setColumnCount(0)


# ─────────────────────────────────────────────────────────────────
#  RUN THREAD
# ─────────────────────────────────────────────────────────────────

class RunWorker(QThread):
    log_signal = pyqtSignal(str, str)
    node_state = pyqtSignal(str, str, str)  # node_id, state, error
    figure_signal = pyqtSignal(object, str)
    dataframe_signal = pyqtSignal(object, str)
    metrics_signal = pyqtSignal(dict, str)
    finished_signal = pyqtSignal()

    def __init__(self, nodes: List[Node], wires: List[Wire]):
        super().__init__()
        self.nodes = nodes
        self.wires = wires
        self.engine = ExecutionEngine()

    def run(self):
        # topological sort
        order = self._topo_sort()
        if order is None:
            self.log_signal.emit("❌ Cycle detected in workflow!", "error")
            return

        for node in order:
            node.state = "running"
            self.node_state.emit(node.id, "running", "")
            try:
                inputs = self._gather_inputs(node)
                result = self.engine.run_block(node.block_name, node.params, inputs)
                node.result = result
                node.state = "ok"
                self.node_state.emit(node.id, "ok", "")
                self.log_signal.emit(f"✅ <b>{node.block_name}</b> — completed", "ok")
                self._emit_result(node, result)
            except Exception as ex:
                node.state = "error"
                node.error_msg = str(ex)
                self.node_state.emit(node.id, "error", str(ex))
                self.log_signal.emit(f"❌ <b>{node.block_name}</b>: {ex}", "error")

        self.finished_signal.emit()

    def _topo_sort(self) -> Optional[List[Node]]:
        """Kahn's algorithm for topological sort."""
        node_map = {n.id: n for n in self.nodes}
        in_deg = {n.id: 0 for n in self.nodes}
        adj = {n.id: [] for n in self.nodes}
        for wire in self.wires:
            u = wire.src.node.id
            v = wire.dst.node.id
            adj[u].append(v)
            in_deg[v] += 1
        queue = [n for n in self.nodes if in_deg[n.id] == 0]
        result = []
        while queue:
            node = queue.pop(0)
            result.append(node)
            for nb_id in adj[node.id]:
                in_deg[nb_id] -= 1
                if in_deg[nb_id] == 0:
                    queue.append(node_map[nb_id])
        return result if len(result) == len(self.nodes) else None

    def _gather_inputs(self, node: Node) -> dict:
        inputs = {}
        for port in node.input_ports:
            for wire in port.connections:
                src_port = wire.src
                src_node = src_port.node
                if src_port.name in src_node.result:
                    inputs[port.name] = src_node.result[src_port.name]
        return inputs

    def _emit_result(self, node: Node, result: dict):
        for key, val in result.items():
            if key == "_figure":
                self.figure_signal.emit(val, f"{node.block_name}")
            elif key == "_metrics":
                self.metrics_signal.emit(val, node.block_name)
            elif key == "_info":
                self.log_signal.emit(f"  ℹ {val}", "info")
            elif isinstance(val, pd.DataFrame):
                self.dataframe_signal.emit(val, node.block_name)
            elif isinstance(val, dict) and "RMSE" in val or isinstance(val, dict) and "Accuracy" in val or isinstance(val, dict) and "CV Mean" in list(val.keys())[:1]:
                self.metrics_signal.emit(val, node.block_name)


# ─────────────────────────────────────────────────────────────────
#  EXAMPLE WORKFLOWS
# ─────────────────────────────────────────────────────────────────

EXAMPLE_WORKFLOWS = {
    "🌸 Iris Classification": {
        "nodes": [
            {"id": "n1", "name": "Sample Dataset", "x": 60, "y": 100,
             "params": {"dataset": "Iris"}},
            {"id": "n2", "name": "Select Features", "x": 320, "y": 100,
             "params": {"target_col": "target", "feature_cols": ""}},
            {"id": "n3", "name": "Train/Test Split", "x": 580, "y": 100,
             "params": {"test_size": 0.2, "random_state": 42, "stratify": True}},
            {"id": "n4", "name": "Random Forest Classifier", "x": 840, "y": 60,
             "params": {"n_estimators": 100, "max_depth": 0, "random_state": 42}},
            {"id": "n5", "name": "Classification Metrics", "x": 1120, "y": 60,
             "params": {}},
            {"id": "n6", "name": "Confusion Matrix", "x": 1120, "y": 200,
             "params": {}},
            {"id": "n7", "name": "ROC Curve", "x": 1120, "y": 340,
             "params": {}},
        ],
        "wires": [
            ("n1", "DataFrame", "n2", "DataFrame"),
            ("n2", "Features", "n3", "Features"),
            ("n2", "Target", "n3", "Target"),
            ("n3", "X_train", "n4", "X_train"),
            ("n3", "y_train", "n4", "y_train"),
            ("n4", "Model", "n5", "Model"),
            ("n3", "X_test", "n5", "X_test"),
            ("n3", "y_test", "n5", "y_test"),
            ("n4", "Model", "n6", "Model"),
            ("n3", "X_test", "n6", "X_test"),
            ("n3", "y_test", "n6", "y_test"),
            ("n4", "Model", "n7", "Model"),
            ("n3", "X_test", "n7", "X_test"),
            ("n3", "y_test", "n7", "y_test"),
        ],
    },
    "📉 Regression Pipeline": {
        "nodes": [
            {"id": "n1", "name": "Sample Dataset", "x": 60, "y": 160,
             "params": {"dataset": "Diabetes"}},
            {"id": "n2", "name": "Impute Missing", "x": 300, "y": 160,
             "params": {"strategy": "mean", "fill_value": "0"}},
            {"id": "n3", "name": "Standard Scaler", "x": 540, "y": 160,
             "params": {"columns": ""}},
            {"id": "n4", "name": "Correlation Heatmap", "x": 540, "y": 320,
             "params": {}},
            {"id": "n5", "name": "Select Features", "x": 780, "y": 160,
             "params": {"target_col": "target", "feature_cols": ""}},
            {"id": "n6", "name": "Train/Test Split", "x": 1020, "y": 160,
             "params": {"test_size": 0.2, "random_state": 42, "stratify": False}},
            {"id": "n7", "name": "Random Forest Regressor", "x": 1260, "y": 80,
             "params": {"n_estimators": 100, "max_depth": 0, "random_state": 42}},
            {"id": "n8", "name": "Regression Metrics", "x": 1500, "y": 80,
             "params": {}},
            {"id": "n9", "name": "Residual Plot", "x": 1500, "y": 240,
             "params": {}},
        ],
        "wires": [
            ("n1", "DataFrame", "n2", "DataFrame"),
            ("n2", "DataFrame", "n3", "DataFrame"),
            ("n3", "DataFrame", "n4", "DataFrame"),
            ("n3", "DataFrame", "n5", "DataFrame"),
            ("n5", "Features", "n6", "Features"),
            ("n5", "Target", "n6", "Target"),
            ("n6", "X_train", "n7", "X_train"),
            ("n6", "y_train", "n7", "y_train"),
            ("n7", "Model", "n8", "Model"),
            ("n6", "X_test", "n8", "X_test"),
            ("n6", "y_test", "n8", "y_test"),
            ("n7", "Model", "n9", "Model"),
            ("n6", "X_test", "n9", "X_test"),
            ("n6", "y_test", "n9", "y_test"),
        ],
    },
    "🔵 Clustering Explorer": {
        "nodes": [
            {"id": "n1", "name": "Sample Dataset", "x": 60, "y": 160,
             "params": {"dataset": "Make Blobs"}},
            {"id": "n2", "name": "Standard Scaler", "x": 300, "y": 160,
             "params": {"columns": ""}},
            {"id": "n3", "name": "Data Summary", "x": 300, "y": 320,
             "params": {}},
            {"id": "n4", "name": "K-Means", "x": 540, "y": 160,
             "params": {"n_clusters": 4, "random_state": 42, "n_init": 10}},
            {"id": "n5", "name": "Scatter Plot", "x": 780, "y": 160,
             "params": {"x_col": "x0", "y_col": "x1", "color_col": "cluster"}},
            {"id": "n6", "name": "Histogram", "x": 780, "y": 320,
             "params": {"column": "", "bins": 20}},
        ],
        "wires": [
            ("n1", "DataFrame", "n2", "DataFrame"),
            ("n2", "DataFrame", "n3", "DataFrame"),
            ("n2", "DataFrame", "n4", "DataFrame"),
            ("n4", "DataFrame", "n5", "DataFrame"),
            ("n4", "DataFrame", "n6", "DataFrame"),
        ],
    },
}


def load_example(name: str, canvas: 'Canvas'):
    wf = EXAMPLE_WORKFLOWS.get(name)
    if not wf:
        return
    canvas.clear_all()
    node_map: Dict[str, Node] = {}
    for nd in wf["nodes"]:
        node = Node(nd["name"], nd["x"], nd["y"])
        node.params.update(nd.get("params", {}))
        node.id = nd["id"]
        canvas.nodes.append(node)
        node_map[nd["id"]] = node
    for src_id, src_port_name, dst_id, dst_port_name in wf["wires"]:
        src_node = node_map[src_id]
        dst_node = node_map[dst_id]
        src_port = next((p for p in src_node.output_ports if p.name == src_port_name), None)
        dst_port = next((p for p in dst_node.input_ports if p.name == dst_port_name), None)
        if src_port and dst_port:
            wire = Wire(src_port, dst_port)
            canvas.wires.append(wire)
    canvas.fit_view()
    canvas.update()


# ─────────────────────────────────────────────────────────────────
#  MAIN WINDOW
# ─────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DataFlow Studio — Visual Data Science Workspace")
        self.resize(1600, 950)
        self._apply_style()
        self._build_ui()
        self._worker: Optional[RunWorker] = None

    def _apply_style(self):
        self.setStyleSheet(f"""
            QMainWindow {{ background: {THEME['bg_dark']}; }}
            /* --- ADD THIS QToolTip SECTION --- */
            QToolTip {{
                background-color: {THEME['bg_panel']};
                color: {THEME['text']};
                border: 1px solid {THEME['border']};
                border-radius: 4px;
                padding: 5px;
                font-size: 13px;
                font-family: 'Segoe UI', sans-serif;
            }}
            /* --------------------------------- */
            QMenuBar {{ background: {THEME['bg_mid']}; color: {THEME['text']}; border-bottom: 1px solid {THEME['border']}; padding: 2px; }}
            QMenuBar::item:selected {{ background: {THEME['accent']}; border-radius: 4px; }}
            QMenu {{ background: {THEME['bg_light']}; color: {THEME['text']}; border: 1px solid {THEME['border']}; border-radius: 6px; padding: 4px; }}
            QMenu::item {{ padding: 6px 20px; }}
            QMenu::item:selected {{ background: {THEME['accent']}; border-radius: 4px; }}
            QSplitter::handle {{ background: {THEME['border']}; }}
            QToolBar {{ background: {THEME['bg_mid']}; border-bottom: 1px solid {THEME['border']}; spacing: 4px; padding: 4px 8px; }}
            QToolButton {{ color: {THEME['text']}; padding: 4px 10px; border-radius: 5px; border: 1px solid transparent; }}
            QToolButton:hover {{ background: {THEME['bg_light']}; border-color: {THEME['border']}; }}
            QStatusBar {{ background: {THEME['bg_mid']}; color: {THEME['text_dim']}; border-top: 1px solid {THEME['border']}; font-size: 11px; }}
        """)

    def _build_ui(self):
        # Menu bar
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        file_menu.addAction("New Workflow", self._new_workflow, QKeySequence("Ctrl+N"))
        file_menu.addAction("Save Workflow", self._save_workflow, QKeySequence("Ctrl+S"))
        file_menu.addAction("Load Workflow", self._load_workflow, QKeySequence("Ctrl+O"))
        file_menu.addSeparator()
        file_menu.addAction("Exit", self.close, QKeySequence("Ctrl+Q"))

        examples_menu = menubar.addMenu("Examples")
        for name in EXAMPLE_WORKFLOWS:
            action = examples_menu.addAction(name)
            action.triggered.connect(lambda checked, n=name: self._load_example(n))

        view_menu = menubar.addMenu("View")
        view_menu.addAction("Fit View", lambda: self.canvas.fit_view(), QKeySequence("F"))
        view_menu.addAction("Reset Zoom", self._reset_zoom)
        view_menu.addAction("Clear Canvas", self.canvas.clear_all if hasattr(self, 'canvas') else lambda: None)

        help_menu = menubar.addMenu("Help")
        help_menu.addAction("Quick Start", self._show_help)
        help_menu.addAction("About", self._show_about)

        # Toolbar
        toolbar = self.addToolBar("Main")
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(18, 18))

        run_btn = QToolButton()
        run_btn.setText("▶  Run All")
        run_btn.setStyleSheet(f"""
            QToolButton {{ background: {THEME['accent2']}; color: white; font-weight: bold;
                          border-radius: 5px; padding: 5px 16px; border: none; font-size: 13px; }}
            QToolButton:hover {{ background: #2ea043; }}
            QToolButton:disabled {{ background: {THEME['bg_light']}; color: {THEME['text_muted']}; }}
        """)
        run_btn.clicked.connect(self._run_workflow)
        toolbar.addWidget(run_btn)
        self._run_btn = run_btn

        toolbar.addSeparator()

        clear_btn = QToolButton()
        clear_btn.setText("🗑  Clear")
        clear_btn.clicked.connect(self._clear_canvas)
        toolbar.addWidget(clear_btn)

        fit_btn = QToolButton()
        fit_btn.setText("⊞  Fit View  [F]")
        fit_btn.clicked.connect(lambda: self.canvas.fit_view())
        toolbar.addWidget(fit_btn)

        toolbar.addSeparator()

        for name in EXAMPLE_WORKFLOWS:
            btn = QToolButton()
            btn.setText(name)
            btn.clicked.connect(lambda checked, n=name: self._load_example(n))
            toolbar.addWidget(btn)

        toolbar.addSeparator()
        self._status_label = QLabel("  Ready")
        self._status_label.setStyleSheet(f"color: {THEME['text_dim']}; font-size: 12px;")
        toolbar.addWidget(self._status_label)

        # Central splitter
        main_splitter = QSplitter(Qt.Horizontal)

        # Left: Block library
        self.library = BlockLibrary()
        self.library.setMinimumWidth(200)
        self.library.setMaximumWidth(280)
        main_splitter.addWidget(self.library)

        # Center: Canvas + output
        center_splitter = QSplitter(Qt.Vertical)

        self.canvas = Canvas()
        self.canvas.node_selected.connect(self._on_node_selected)
        self.canvas.run_requested.connect(self._run_workflow)
        center_splitter.addWidget(self.canvas)

        self.result_viewer = ResultViewer()
        self.result_viewer.setMinimumHeight(180)
        center_splitter.addWidget(self.result_viewer)
        center_splitter.setSizes([600, 300])

        main_splitter.addWidget(center_splitter)

        # Right: Param editor
        self.param_editor = ParamEditor()
        self.param_editor.setMinimumWidth(220)
        self.param_editor.setMaximumWidth(320)
        main_splitter.addWidget(self.param_editor)

        main_splitter.setSizes([240, 1100, 260])
        self.setCentralWidget(main_splitter)

        # Status bar
        self.statusBar().showMessage("DataFlow Studio ready — drag blocks from the library to start building your workflow")

        # Load default example
        QTimer.singleShot(100, lambda: self._load_example("🌸 Iris Classification"))

    def _on_node_selected(self, node: Optional[Node]):
        self.param_editor.set_node(node)

    def _run_workflow(self):
        if not self.canvas.nodes:
            QMessageBox.information(self, "Empty Canvas", "Add some blocks to the canvas first!")
            return
        if self._worker and self._worker.isRunning():
            return

        # Reset states
        for node in self.canvas.nodes:
            node.state = "idle"
            node.result = {}
        self.canvas.update()

        self.result_viewer.clear()
        self.result_viewer.log("🚀 Starting workflow execution...", "info")
        self._run_btn.setEnabled(False)
        self._status_label.setText("  ⏳ Running...")

        self._worker = RunWorker(list(self.canvas.nodes), list(self.canvas.wires))
        self._worker.log_signal.connect(self.result_viewer.log)
        self._worker.node_state.connect(self._on_node_state)
        self._worker.figure_signal.connect(self.result_viewer.show_figure)
        self._worker.dataframe_signal.connect(self.result_viewer.show_dataframe)
        self._worker.metrics_signal.connect(self.result_viewer.show_metrics)
        self._worker.finished_signal.connect(self._on_run_finished)
        self._worker.start()

    def _on_node_state(self, node_id: str, state: str, error: str):
        for node in self.canvas.nodes:
            if node.id == node_id:
                node.state = state
                if error:
                    node.error_msg = error
                break
        self.canvas.update()

    def _on_run_finished(self):
        self._run_btn.setEnabled(True)
        errors = [n for n in self.canvas.nodes if n.state == "error"]
        ok_nodes = [n for n in self.canvas.nodes if n.state == "ok"]
        if errors:
            self._status_label.setText(f"  ❌ {len(errors)} error(s)")
            self.result_viewer.log(f"\n⚠️  Workflow completed with {len(errors)} error(s). "
                                   f"{len(ok_nodes)} nodes succeeded.", "warn")
        else:
            self._status_label.setText(f"  ✅ Done — {len(ok_nodes)} nodes")
            self.result_viewer.log(f"\n✅ Workflow completed successfully! ({len(ok_nodes)} nodes)", "ok")

    def _clear_canvas(self):
        reply = QMessageBox.question(self, "Clear Canvas", "Clear all nodes and connections?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.canvas.clear_all()
            self.result_viewer.clear()
            self._status_label.setText("  Ready")

    def _load_example(self, name: str):
        load_example(name, self.canvas)
        self.result_viewer.clear()
        self.result_viewer.log(f"📂 Loaded example: <b>{name}</b>", "info")
        self.result_viewer.log("  Click <b>▶ Run All</b> to execute the workflow.", "info")
        self._status_label.setText(f"  Loaded: {name}")

    def _reset_zoom(self):
        self.canvas.scale = 1.0
        self.canvas.offset = QPointF(0, 0)
        self.canvas.update()

    def _new_workflow(self):
        self._clear_canvas()

    def _save_workflow(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Workflow", "workflow.json", "JSON (*.json)")
        if not path:
            return
        data = {
            "nodes": [{"id": n.id, "name": n.block_name,
                       "x": n.rect.x(), "y": n.rect.y(), "params": n.params}
                      for n in self.canvas.nodes],
            "wires": [{"src_node": w.src.node.id, "src_port": w.src.name,
                       "dst_node": w.dst.node.id, "dst_port": w.dst.name}
                      for w in self.canvas.wires],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        self.statusBar().showMessage(f"Workflow saved to {path}")

    def _load_workflow(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Workflow", "", "JSON (*.json)")
        if not path:
            return
        with open(path) as f:
            data = json.load(f)
        self.canvas.clear_all()
        node_map: Dict[str, Node] = {}
        for nd in data.get("nodes", []):
            if nd["name"] not in BLOCK_DEFS:
                continue
            node = Node(nd["name"], nd["x"], nd["y"])
            node.id = nd["id"]
            node.params.update(nd.get("params", {}))
            self.canvas.nodes.append(node)
            node_map[nd["id"]] = node
        for wd in data.get("wires", []):
            src_node = node_map.get(wd["src_node"])
            dst_node = node_map.get(wd["dst_node"])
            if not src_node or not dst_node:
                continue
            src_port = next((p for p in src_node.output_ports if p.name == wd["src_port"]), None)
            dst_port = next((p for p in dst_node.input_ports if p.name == wd["dst_port"]), None)
            if src_port and dst_port:
                wire = Wire(src_port, dst_port)
                self.canvas.wires.append(wire)
        self.canvas.fit_view()
        self.canvas.update()
        self.statusBar().showMessage(f"Workflow loaded from {path}")

    def _show_help(self):
        msg = QMessageBox(self)
        msg.setWindowTitle("Quick Start — DataFlow Studio")
        msg.setText("""
<h3 style='color:#58A6FF;'>Welcome to DataFlow Studio</h3>
<b>Building a Workflow:</b>
<ul>
<li><b>Drag</b> blocks from the left library onto the canvas</li>
<li><b>Connect</b> ports by clicking & dragging from an output (right side of block) to an input (left side)</li>
<li><b>Configure</b> blocks by clicking them and editing parameters in the right panel</li>
<li><b>Run</b> the workflow with the ▶ Run All button</li>
</ul>
<b>Navigation:</b>
<ul>
<li><b>Scroll</b> to zoom in/out</li>
<li><b>Middle-click drag</b> or Alt+drag to pan</li>
<li><b>Delete</b> key removes selected nodes</li>
<li><b>F</b> key fits all nodes in view</li>
<li><b>Ctrl+A</b> selects all nodes</li>
</ul>
<b>Tips:</b>
<ul>
<li>Load example workflows from the toolbar or Examples menu</li>
<li>Save/load workflows as JSON files</li>
<li>Right-click for context menu options</li>
</ul>
        """)
        msg.setStyleSheet(f"QMessageBox {{ background: {THEME['bg_mid']}; color: {THEME['text']}; }}")
        msg.exec_()

    def _show_about(self):
        QMessageBox.about(self, "About DataFlow Studio",
                          "DataFlow Studio v1.0\n\nA visual data science workflow builder.\n\n"
                          "Built with PyQt5 + scikit-learn + pandas + matplotlib ")


# ─────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("DataFlow Studio")
    app.setStyle("Fusion")

    # Dark palette
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(THEME["bg_dark"]))
    palette.setColor(QPalette.WindowText, QColor(THEME["text"]))
    palette.setColor(QPalette.Base, QColor(THEME["bg_mid"]))
    palette.setColor(QPalette.AlternateBase, QColor(THEME["bg_light"]))
    palette.setColor(QPalette.Text, QColor(THEME["text"]))
    palette.setColor(QPalette.Button, QColor(THEME["bg_light"]))
    palette.setColor(QPalette.ButtonText, QColor(THEME["text"]))
    palette.setColor(QPalette.Highlight, QColor(THEME["accent"]))
    palette.setColor(QPalette.HighlightedText, Qt.white)
    app.setPalette(palette)

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
