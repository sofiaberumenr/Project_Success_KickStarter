import json

with open('kickstarter_analysis.ipynb', 'r') as f:
    nb = json.load(f)

# Find where to insert after Random Forest
insert_idx = None
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and 'Random Forest' in ''.join(cell['source']):
        insert_idx = i + 1
        break

remaining_cells = [
    # Gradient Boosting
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 4. Gradient Boosting\n",
            "gb_model = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)\n",
            "gb_model.fit(X_train, y_train)\n",
            "\n",
            "y_pred_gb = gb_model.predict(X_test)\n",
            "y_pred_proba_gb = gb_model.predict_proba(X_test)[:, 1]\n",
            "\n",
            "print(\"\\n=== Gradient Boosting ===\")\n",
            "print(f\"Accuracy: {accuracy_score(y_test, y_pred_gb):.4f}\")\n",
            "print(f\"Precision: {precision_score(y_test, y_pred_gb):.4f}\")\n",
            "print(f\"Recall: {recall_score(y_test, y_pred_gb):.4f}\")\n",
            "print(f\"F1-Score: {f1_score(y_test, y_pred_gb):.4f}\")\n",
            "print(f\"ROC-AUC: {roc_auc_score(y_test, y_pred_proba_gb):.4f}\")"
        ]
    },
    # KNN
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 5. K-Nearest Neighbors\n",
            "knn_model = KNeighborsClassifier(n_neighbors=5)\n",
            "knn_model.fit(X_train_scaled, y_train)\n",
            "\n",
            "y_pred_knn = knn_model.predict(X_test_scaled)\n",
            "y_pred_proba_knn = knn_model.predict_proba(X_test_scaled)[:, 1]\n",
            "\n",
            "print(\"\\n=== K-Nearest Neighbors ===\")\n",
            "print(f\"Accuracy: {accuracy_score(y_test, y_pred_knn):.4f}\")\n",
            "print(f\"Precision: {precision_score(y_test, y_pred_knn):.4f}\")\n",
            "print(f\"Recall: {recall_score(y_test, y_pred_knn):.4f}\")\n",
            "print(f\"F1-Score: {f1_score(y_test, y_pred_knn):.4f}\")\n",
            "print(f\"ROC-AUC: {roc_auc_score(y_test, y_pred_proba_knn):.4f}\")"
        ]
    },
    # SVM
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 6. Support Vector Machine (on sample for speed)\n",
            "svm_model = SVC(kernel='rbf', probability=True, random_state=42)\n",
            "sample_size = min(10000, len(X_train_scaled))\n",
            "sample_idx = np.random.choice(len(X_train_scaled), sample_size, replace=False)\n",
            "svm_model.fit(X_train_scaled[sample_idx], y_train.iloc[sample_idx])\n",
            "\n",
            "y_pred_svm = svm_model.predict(X_test_scaled)\n",
            "y_pred_proba_svm = svm_model.predict_proba(X_test_scaled)[:, 1]\n",
            "\n",
            "print(\"\\n=== Support Vector Machine ===\")\n",
            "print(f\"Accuracy: {accuracy_score(y_test, y_pred_svm):.4f}\")\n",
            "print(f\"Precision: {precision_score(y_test, y_pred_svm):.4f}\")\n",
            "print(f\"Recall: {recall_score(y_test, y_pred_svm):.4f}\")\n",
            "print(f\"F1-Score: {f1_score(y_test, y_pred_svm):.4f}\")\n",
            "print(f\"ROC-AUC: {roc_auc_score(y_test, y_pred_proba_svm):.4f}\")"
        ]
    },
    # Naive Bayes
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 7. Naive Bayes\n",
            "from sklearn.naive_bayes import GaussianNB\n",
            "\n",
            "nb_model = GaussianNB()\n",
            "nb_model.fit(X_train_scaled, y_train)\n",
            "\n",
            "y_pred_nb = nb_model.predict(X_test_scaled)\n",
            "y_pred_proba_nb = nb_model.predict_proba(X_test_scaled)[:, 1]\n",
            "\n",
            "print(\"\\n=== Naive Bayes ===\")\n",
            "print(f\"Accuracy: {accuracy_score(y_test, y_pred_nb):.4f}\")\n",
            "print(f\"Precision: {precision_score(y_test, y_pred_nb):.4f}\")\n",
            "print(f\"Recall: {recall_score(y_test, y_pred_nb):.4f}\")\n",
            "print(f\"F1-Score: {f1_score(y_test, y_pred_nb):.4f}\")\n",
            "print(f\"ROC-AUC: {roc_auc_score(y_test, y_pred_proba_nb):.4f}\")"
        ]
    },
    # Model Comparison
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["### Model Comparison"]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Compare all models\n",
            "results = pd.DataFrame({\n",
            "    'Model': ['Logistic Regression', 'Decision Tree', 'Random Forest', \n",
            "              'Gradient Boosting', 'KNN', 'SVM', 'Naive Bayes'],\n",
            "    'Accuracy': [accuracy_score(y_test, y_pred_lr), accuracy_score(y_test, y_pred_dt),\n",
            "                 accuracy_score(y_test, y_pred_rf), accuracy_score(y_test, y_pred_gb),\n",
            "                 accuracy_score(y_test, y_pred_knn), accuracy_score(y_test, y_pred_svm),\n",
            "                 accuracy_score(y_test, y_pred_nb)],\n",
            "    'F1-Score': [f1_score(y_test, y_pred_lr), f1_score(y_test, y_pred_dt),\n",
            "                 f1_score(y_test, y_pred_rf), f1_score(y_test, y_pred_gb),\n",
            "                 f1_score(y_test, y_pred_knn), f1_score(y_test, y_pred_svm),\n",
            "                 f1_score(y_test, y_pred_nb)],\n",
            "    'ROC-AUC': [roc_auc_score(y_test, y_pred_proba_lr), roc_auc_score(y_test, y_pred_proba_dt),\n",
            "                roc_auc_score(y_test, y_pred_proba_rf), roc_auc_score(y_test, y_pred_proba_gb),\n",
            "                roc_auc_score(y_test, y_pred_proba_knn), roc_auc_score(y_test, y_pred_proba_svm),\n",
            "                roc_auc_score(y_test, y_pred_proba_nb)]\n",
            "})\n",
            "\n",
            "print(\"\\n=== Model Comparison ===\")\n",
            "print(results.sort_values('ROC-AUC', ascending=False))\n",
            "\n",
            "# Visualize\n",
            "fig, ax = plt.subplots(figsize=(12, 6))\n",
            "results.plot(x='Model', y=['Accuracy', 'F1-Score', 'ROC-AUC'], kind='bar', ax=ax, rot=45)\n",
            "ax.set_title('Model Performance Comparison')\n",
            "ax.set_ylabel('Score')\n",
            "ax.legend(loc='lower right')\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    },
    # Task 2
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Task 2 – Clustering Model"]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Prepare data for clustering\n",
            "X_clustering = X_train_scaled.copy()\n",
            "sample_size = min(20000, len(X_clustering))\n",
            "sample_idx = np.random.choice(len(X_clustering), sample_size, replace=False)\n",
            "X_sample = X_clustering[sample_idx]\n",
            "print(f\"Clustering on {len(X_sample)} samples\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# K-Means - Find optimal k\n",
            "inertias = []\n",
            "silhouette_scores = []\n",
            "K_range = range(2, 11)\n",
            "\n",
            "for k in K_range:\n",
            "    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)\n",
            "    kmeans.fit(X_sample)\n",
            "    inertias.append(kmeans.inertia_)\n",
            "    silhouette_scores.append(silhouette_score(X_sample, kmeans.labels_))\n",
            "\n",
            "# Plot\n",
            "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
            "axes[0].plot(K_range, inertias, 'bo-')\n",
            "axes[0].set_xlabel('Number of Clusters (k)')\n",
            "axes[0].set_ylabel('Inertia')\n",
            "axes[0].set_title('Elbow Method')\n",
            "axes[1].plot(K_range, silhouette_scores, 'ro-')\n",
            "axes[1].set_xlabel('Number of Clusters (k)')\n",
            "axes[1].set_ylabel('Silhouette Score')\n",
            "axes[1].set_title('Silhouette Score vs K')\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Fit K-Means and Hierarchical\n",
            "optimal_k = 4\n",
            "kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)\n",
            "kmeans_labels = kmeans_final.fit_predict(X_sample)\n",
            "\n",
            "hierarchical = AgglomerativeClustering(n_clusters=optimal_k)\n",
            "hierarchical_labels = hierarchical.fit_predict(X_sample)\n",
            "\n",
            "print(f\"K-Means Silhouette: {silhouette_score(X_sample, kmeans_labels):.4f}\")\n",
            "print(f\"Hierarchical Silhouette: {silhouette_score(X_sample, hierarchical_labels):.4f}\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# PCA Visualization\n",
            "pca = PCA(n_components=2, random_state=42)\n",
            "X_pca = pca.fit_transform(X_sample)\n",
            "\n",
            "fig, axes = plt.subplots(1, 2, figsize=(16, 6))\n",
            "axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap='viridis', alpha=0.6, s=20)\n",
            "axes[0].set_title('K-Means Clustering - PCA')\n",
            "axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=hierarchical_labels, cmap='plasma', alpha=0.6, s=20)\n",
            "axes[1].set_title('Hierarchical Clustering - PCA')\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    }
]

for i, cell in enumerate(remaining_cells):
    nb['cells'].insert(insert_idx + i, cell)

with open('kickstarter_analysis.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print(f"Added {len(remaining_cells)} cells")
