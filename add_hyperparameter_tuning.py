import json

with open('kickstarter_analysis.ipynb', 'r') as f:
    nb = json.load(f)

# Find where to insert (after Model Comparison section)
insert_idx = None
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown' and 'Model Comparison' in ''.join(cell['source']):
        # Find the next code cell after comparison
        for j in range(i+1, len(nb['cells'])):
            if nb['cells'][j]['cell_type'] == 'code':
                insert_idx = j + 1
                break
        break

if not insert_idx:
    print("Could not find insertion point")
    exit(1)

tuning_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Hyperparameter Tuning\n",
            "\n",
            "Tuning the top 4 models using RandomizedSearchCV with 3-fold cross-validation."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from sklearn.model_selection import RandomizedSearchCV, GridSearchCV\n",
            "from scipy.stats import randint, uniform\n",
            "import time\n",
            "\n",
            "# Use a sample for faster tuning\n",
            "tune_sample_size = min(30000, len(X_train))\n",
            "tune_idx = np.random.choice(len(X_train), tune_sample_size, replace=False)\n",
            "X_train_tune = X_train.iloc[tune_idx]\n",
            "y_train_tune = y_train.iloc[tune_idx]\n",
            "X_train_tune_scaled = X_train_scaled[tune_idx]\n",
            "\n",
            "print(f\"Tuning on {tune_sample_size} samples with 3-fold CV\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["### 1. Logistic Regression Tuning"]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Logistic Regression hyperparameter tuning\n",
            "lr_param_grid = {\n",
            "    'C': [0.001, 0.01, 0.1, 1, 10, 100],\n",
            "    'penalty': ['l1', 'l2'],\n",
            "    'solver': ['liblinear', 'saga'],\n",
            "    'max_iter': [1000]\n",
            "}\n",
            "\n",
            "print(\"Tuning Logistic Regression...\")\n",
            "start_time = time.time()\n",
            "\n",
            "lr_grid = GridSearchCV(\n",
            "    LogisticRegression(random_state=42),\n",
            "    lr_param_grid,\n",
            "    cv=3,\n",
            "    scoring='roc_auc',\n",
            "    n_jobs=-1,\n",
            "    verbose=1\n",
            ")\n",
            "lr_grid.fit(X_train_tune_scaled, y_train_tune)\n",
            "\n",
            "print(f\"\\nBest parameters: {lr_grid.best_params_}\")\n",
            "print(f\"Best CV score: {lr_grid.best_score_:.4f}\")\n",
            "print(f\"Time taken: {time.time() - start_time:.2f} seconds\")\n",
            "\n",
            "# Evaluate on test set\n",
            "lr_tuned = lr_grid.best_estimator_\n",
            "y_pred_lr_tuned = lr_tuned.predict(X_test_scaled)\n",
            "y_pred_proba_lr_tuned = lr_tuned.predict_proba(X_test_scaled)[:, 1]\n",
            "\n",
            "print(f\"\\nTest Accuracy: {accuracy_score(y_test, y_pred_lr_tuned):.4f}\")\n",
            "print(f\"Test ROC-AUC: {roc_auc_score(y_test, y_pred_proba_lr_tuned):.4f}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["### 2. Random Forest Tuning"]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Random Forest hyperparameter tuning\n",
            "rf_param_dist = {\n",
            "    'n_estimators': [50, 100, 200],\n",
            "    'max_depth': [5, 10, 15, 20, None],\n",
            "    'min_samples_split': [2, 5, 10],\n",
            "    'min_samples_leaf': [1, 2, 4],\n",
            "    'max_features': ['sqrt', 'log2'],\n",
            "    'bootstrap': [True, False]\n",
            "}\n",
            "\n",
            "print(\"Tuning Random Forest...\")\n",
            "start_time = time.time()\n",
            "\n",
            "rf_random = RandomizedSearchCV(\n",
            "    RandomForestClassifier(random_state=42, n_jobs=-1),\n",
            "    rf_param_dist,\n",
            "    n_iter=20,  # Number of parameter settings sampled\n",
            "    cv=3,\n",
            "    scoring='roc_auc',\n",
            "    n_jobs=-1,\n",
            "    verbose=1,\n",
            "    random_state=42\n",
            ")\n",
            "rf_random.fit(X_train_tune, y_train_tune)\n",
            "\n",
            "print(f\"\\nBest parameters: {rf_random.best_params_}\")\n",
            "print(f\"Best CV score: {rf_random.best_score_:.4f}\")\n",
            "print(f\"Time taken: {time.time() - start_time:.2f} seconds\")\n",
            "\n",
            "# Evaluate on test set\n",
            "rf_tuned = rf_random.best_estimator_\n",
            "y_pred_rf_tuned = rf_tuned.predict(X_test)\n",
            "y_pred_proba_rf_tuned = rf_tuned.predict_proba(X_test)[:, 1]\n",
            "\n",
            "print(f\"\\nTest Accuracy: {accuracy_score(y_test, y_pred_rf_tuned):.4f}\")\n",
            "print(f\"Test ROC-AUC: {roc_auc_score(y_test, y_pred_proba_rf_tuned):.4f}\")\n",
            "\n",
            "# Feature importance from tuned model\n",
            "feature_importance_tuned = pd.DataFrame({\n",
            "    'feature': X_train.columns,\n",
            "    'importance': rf_tuned.feature_importances_\n",
            "}).sort_values('importance', ascending=False)\n",
            "\n",
            "print(\"\\nTop 10 Important Features (Tuned Model):\")\n",
            "print(feature_importance_tuned.head(10))"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["### 3. Gradient Boosting Tuning"]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Gradient Boosting hyperparameter tuning\n",
            "gb_param_dist = {\n",
            "    'n_estimators': [50, 100, 150],\n",
            "    'learning_rate': [0.01, 0.05, 0.1, 0.2],\n",
            "    'max_depth': [3, 4, 5, 6],\n",
            "    'min_samples_split': [2, 5, 10],\n",
            "    'min_samples_leaf': [1, 2, 4],\n",
            "    'subsample': [0.8, 0.9, 1.0],\n",
            "    'max_features': ['sqrt', 'log2', None]\n",
            "}\n",
            "\n",
            "print(\"Tuning Gradient Boosting...\")\n",
            "start_time = time.time()\n",
            "\n",
            "gb_random = RandomizedSearchCV(\n",
            "    GradientBoostingClassifier(random_state=42),\n",
            "    gb_param_dist,\n",
            "    n_iter=20,\n",
            "    cv=3,\n",
            "    scoring='roc_auc',\n",
            "    n_jobs=-1,\n",
            "    verbose=1,\n",
            "    random_state=42\n",
            ")\n",
            "gb_random.fit(X_train_tune, y_train_tune)\n",
            "\n",
            "print(f\"\\nBest parameters: {gb_random.best_params_}\")\n",
            "print(f\"Best CV score: {gb_random.best_score_:.4f}\")\n",
            "print(f\"Time taken: {time.time() - start_time:.2f} seconds\")\n",
            "\n",
            "# Evaluate on test set\n",
            "gb_tuned = gb_random.best_estimator_\n",
            "y_pred_gb_tuned = gb_tuned.predict(X_test)\n",
            "y_pred_proba_gb_tuned = gb_tuned.predict_proba(X_test)[:, 1]\n",
            "\n",
            "print(f\"\\nTest Accuracy: {accuracy_score(y_test, y_pred_gb_tuned):.4f}\")\n",
            "print(f\"Test ROC-AUC: {roc_auc_score(y_test, y_pred_proba_gb_tuned):.4f}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["### 4. K-Nearest Neighbors Tuning"]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# KNN hyperparameter tuning\n",
            "knn_param_grid = {\n",
            "    'n_neighbors': [3, 5, 7, 9, 11, 15],\n",
            "    'weights': ['uniform', 'distance'],\n",
            "    'metric': ['euclidean', 'manhattan', 'minkowski'],\n",
            "    'p': [1, 2]\n",
            "}\n",
            "\n",
            "print(\"Tuning K-Nearest Neighbors...\")\n",
            "start_time = time.time()\n",
            "\n",
            "knn_grid = GridSearchCV(\n",
            "    KNeighborsClassifier(),\n",
            "    knn_param_grid,\n",
            "    cv=3,\n",
            "    scoring='roc_auc',\n",
            "    n_jobs=-1,\n",
            "    verbose=1\n",
            ")\n",
            "knn_grid.fit(X_train_tune_scaled, y_train_tune)\n",
            "\n",
            "print(f\"\\nBest parameters: {knn_grid.best_params_}\")\n",
            "print(f\"Best CV score: {knn_grid.best_score_:.4f}\")\n",
            "print(f\"Time taken: {time.time() - start_time:.2f} seconds\")\n",
            "\n",
            "# Evaluate on test set\n",
            "knn_tuned = knn_grid.best_estimator_\n",
            "y_pred_knn_tuned = knn_tuned.predict(X_test_scaled)\n",
            "y_pred_proba_knn_tuned = knn_tuned.predict_proba(X_test_scaled)[:, 1]\n",
            "\n",
            "print(f\"\\nTest Accuracy: {accuracy_score(y_test, y_pred_knn_tuned):.4f}\")\n",
            "print(f\"Test ROC-AUC: {roc_auc_score(y_test, y_pred_proba_knn_tuned):.4f}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["### Comparison: Before vs After Tuning"]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Compare original vs tuned models\n",
            "comparison = pd.DataFrame({\n",
            "    'Model': ['Logistic Regression', 'Random Forest', 'Gradient Boosting', 'KNN'],\n",
            "    'Original Test Accuracy': [\n",
            "        accuracy_score(y_test, y_pred_lr),\n",
            "        accuracy_score(y_test, y_pred_rf),\n",
            "        accuracy_score(y_test, y_pred_gb),\n",
            "        accuracy_score(y_test, y_pred_knn)\n",
            "    ],\n",
            "    'Tuned Test Accuracy': [\n",
            "        accuracy_score(y_test, y_pred_lr_tuned),\n",
            "        accuracy_score(y_test, y_pred_rf_tuned),\n",
            "        accuracy_score(y_test, y_pred_gb_tuned),\n",
            "        accuracy_score(y_test, y_pred_knn_tuned)\n",
            "    ],\n",
            "    'Original Test ROC-AUC': [\n",
            "        roc_auc_score(y_test, y_pred_proba_lr),\n",
            "        roc_auc_score(y_test, y_pred_proba_rf),\n",
            "        roc_auc_score(y_test, y_pred_proba_gb),\n",
            "        roc_auc_score(y_test, y_pred_proba_knn)\n",
            "    ],\n",
            "    'Tuned Test ROC-AUC': [\n",
            "        roc_auc_score(y_test, y_pred_proba_lr_tuned),\n",
            "        roc_auc_score(y_test, y_pred_proba_rf_tuned),\n",
            "        roc_auc_score(y_test, y_pred_proba_gb_tuned),\n",
            "        roc_auc_score(y_test, y_pred_proba_knn_tuned)\n",
            "    ]\n",
            "})\n",
            "\n",
            "comparison['Accuracy Improvement'] = comparison['Tuned Test Accuracy'] - comparison['Original Test Accuracy']\n",
            "comparison['ROC-AUC Improvement'] = comparison['Tuned Test ROC-AUC'] - comparison['Original Test ROC-AUC']\n",
            "\n",
            "print(\"\\n=== Before vs After Tuning ===\")\n",
            "print(comparison)\n",
            "\n",
            "# Visualize improvement\n",
            "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
            "\n",
            "x = np.arange(len(comparison['Model']))\n",
            "width = 0.35\n",
            "\n",
            "# Accuracy comparison\n",
            "axes[0].bar(x - width/2, comparison['Original Test Accuracy'], width, label='Original', alpha=0.8)\n",
            "axes[0].bar(x + width/2, comparison['Tuned Test Accuracy'], width, label='Tuned', alpha=0.8)\n",
            "axes[0].set_xlabel('Model')\n",
            "axes[0].set_ylabel('Test Accuracy')\n",
            "axes[0].set_title('Test Accuracy: Original vs Tuned')\n",
            "axes[0].set_xticks(x)\n",
            "axes[0].set_xticklabels(comparison['Model'], rotation=45, ha='right')\n",
            "axes[0].legend()\n",
            "axes[0].grid(axis='y', alpha=0.3)\n",
            "\n",
            "# ROC-AUC comparison\n",
            "axes[1].bar(x - width/2, comparison['Original Test ROC-AUC'], width, label='Original', alpha=0.8)\n",
            "axes[1].bar(x + width/2, comparison['Tuned Test ROC-AUC'], width, label='Tuned', alpha=0.8)\n",
            "axes[1].set_xlabel('Model')\n",
            "axes[1].set_ylabel('Test ROC-AUC')\n",
            "axes[1].set_title('Test ROC-AUC: Original vs Tuned')\n",
            "axes[1].set_xticks(x)\n",
            "axes[1].set_xticklabels(comparison['Model'], rotation=45, ha='right')\n",
            "axes[1].legend()\n",
            "axes[1].grid(axis='y', alpha=0.3)\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()\n",
            "\n",
            "print(\"\\n=== Best Tuned Model ===\")\n",
            "best_model_idx = comparison['Tuned Test ROC-AUC'].idxmax()\n",
            "print(f\"Model: {comparison.loc[best_model_idx, 'Model']}\")\n",
            "print(f\"Test Accuracy: {comparison.loc[best_model_idx, 'Tuned Test Accuracy']:.4f}\")\n",
            "print(f\"Test ROC-AUC: {comparison.loc[best_model_idx, 'Tuned Test ROC-AUC']:.4f}\")"
        ]
    }
]

# Insert all cells
for i, cell in enumerate(tuning_cells):
    nb['cells'].insert(insert_idx + i, cell)

with open('kickstarter_analysis.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print(f"Added {len(tuning_cells)} hyperparameter tuning cells")
