import json

# Read the notebook
with open('kickstarter_analysis.ipynb', 'r') as f:
    nb = json.load(f)

# Find the index where we need to insert (after "Classification Models" markdown)
insert_idx = None
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown' and 'Classification Models' in ''.join(cell['source']):
        insert_idx = i + 1
        break

if insert_idx is None:
    print("Could not find insertion point")
    exit(1)

# Define all model cells
model_cells = [
    # Logistic Regression
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 1. Logistic Regression\n",
            "lr_model = LogisticRegression(max_iter=1000, random_state=42)\n",
            "lr_model.fit(X_train_scaled, y_train)\n",
            "\n",
            "y_pred_lr = lr_model.predict(X_test_scaled)\n",
            "y_pred_proba_lr = lr_model.predict_proba(X_test_scaled)[:, 1]\n",
            "\n",
            "print(\"=== Logistic Regression ===\")\n",
            "print(f\"Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}\")\n",
            "print(f\"Precision: {precision_score(y_test, y_pred_lr):.4f}\")\n",
            "print(f\"Recall: {recall_score(y_test, y_pred_lr):.4f}\")\n",
            "print(f\"F1-Score: {f1_score(y_test, y_pred_lr):.4f}\")\n",
            "print(f\"ROC-AUC: {roc_auc_score(y_test, y_pred_proba_lr):.4f}\")\n",
            "print(\"\\nConfusion Matrix:\")\n",
            "print(confusion_matrix(y_test, y_pred_lr))"
        ]
    },
    # Decision Tree
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 2. Decision Tree\n",
            "dt_model = DecisionTreeClassifier(max_depth=10, random_state=42)\n",
            "dt_model.fit(X_train, y_train)\n",
            "\n",
            "y_pred_dt = dt_model.predict(X_test)\n",
            "y_pred_proba_dt = dt_model.predict_proba(X_test)[:, 1]\n",
            "\n",
            "print(\"\\n=== Decision Tree ===\")\n",
            "print(f\"Accuracy: {accuracy_score(y_test, y_pred_dt):.4f}\")\n",
            "print(f\"Precision: {precision_score(y_test, y_pred_dt):.4f}\")\n",
            "print(f\"Recall: {recall_score(y_test, y_pred_dt):.4f}\")\n",
            "print(f\"F1-Score: {f1_score(y_test, y_pred_dt):.4f}\")\n",
            "print(f\"ROC-AUC: {roc_auc_score(y_test, y_pred_proba_dt):.4f}\")"
        ]
    },
    # Random Forest
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 3. Random Forest\n",
            "rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)\n",
            "rf_model.fit(X_train, y_train)\n",
            "\n",
            "y_pred_rf = rf_model.predict(X_test)\n",
            "y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]\n",
            "\n",
            "print(\"\\n=== Random Forest ===\")\n",
            "print(f\"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}\")\n",
            "print(f\"Precision: {precision_score(y_test, y_pred_rf):.4f}\")\n",
            "print(f\"Recall: {recall_score(y_test, y_pred_rf):.4f}\")\n",
            "print(f\"F1-Score: {f1_score(y_test, y_pred_rf):.4f}\")\n",
            "print(f\"ROC-AUC: {roc_auc_score(y_test, y_pred_proba_rf):.4f}\")\n",
            "\n",
            "# Feature importance\n",
            "feature_importance = pd.DataFrame({\n",
            "    'feature': X_train.columns,\n",
            "    'importance': rf_model.feature_importances_\n",
            "}).sort_values('importance', ascending=False)\n",
            "\n",
            "print(\"\\nTop 10 Important Features:\")\n",
            "print(feature_importance.head(10))"
        ]
    }
]

# Insert cells
for i, cell in enumerate(model_cells):
    nb['cells'].insert(insert_idx + i, cell)

# Save the notebook
with open('kickstarter_analysis.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print(f"Added {len(model_cells)} cells at index {insert_idx}")
