import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from custom_DT import CustomFairDecisionTreeClassifier
import matplotlib.pyplot as plt
from custom_evaluation import calculate_discrimination_score, calculate_fairness_metrics


if __name__ == '__main__':
    # ====================================================================================================
    # ====================================== Preprocess the dataset
    # ====================================================================================================
    file_path = "adult.data"
    column_names = [
        "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
        "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
        "hours-per-week", "native-country", "income"
    ]

    adult = pd.read_csv(file_path, names=column_names, na_values=" ?", skipinitialspace=True)
    adult = adult.dropna()

    # Encode categorical variables
    le = LabelEncoder()
    for col in adult.columns:
        if adult[col].dtype == 'object':
            adult[col] = le.fit_transform(adult[col])

    # Split features and target
    X = adult.drop("income", axis=1)
    y = adult["income"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.3, random_state=42)



    # ====================================================================================================
    # =========================================================== Fair DT With FIG
    # ====================================================================================================
    # index of the protected (sensitive) attribute
    # protected_attributes = ["age", "sex", "race"]
    # protected_attributes = ["age", "race"]
    protected_attributes = ["sex"]
    # protected_attributes = []
    protected_indices = [X.columns.get_loc(attr) for attr in protected_attributes]

    custom_clf = CustomFairDecisionTreeClassifier(max_depth=5)
    custom_clf.fit(X_train, y_train, protected_indices)
    y_pred_custom = custom_clf.predict(X_test)

    # Metrics
    accuracy_custom = accuracy_score(y_test, y_pred_custom)
    precision_custom = precision_score(y_test, y_pred_custom)
    recall_custom = recall_score(y_test, y_pred_custom)
    f1_custom = f1_score(y_test, y_pred_custom)
    cm_custom = confusion_matrix(y_test, y_pred_custom)

    # Evaluate the model
    print("Fair Decision Tree Accuracy:", accuracy_custom)
    print("Fair Decision Tree Precision:", precision_custom)
    print("Fair Decision Tree Recall:", recall_custom)
    print("Fair Decision Tree F1-score:", f1_custom)
    print("Fair Decision Tree Confusion Matrix:\n", cm_custom)


    # ====================================================================================================
    # =========================================================== Traditional DT Without FIG
    # ====================================================================================================
    # index of the protected (sensitive) attribute
    protected_attributes = []
    protected_indices = [X.columns.get_loc(attr) for attr in protected_attributes]

    traditional_clf = CustomFairDecisionTreeClassifier(max_depth=5)
    traditional_clf.fit(X_train, y_train, protected_indices)
    y_pred = traditional_clf.predict(X_test)

    # Metrics
    accuracy_traditional = accuracy_score(y_test, y_pred)
    precision_traditional = precision_score(y_test, y_pred)
    recall_traditional = recall_score(y_test, y_pred)
    f1_traditional = f1_score(y_test, y_pred)
    cm_traditional = confusion_matrix(y_test, y_pred)

    # Evaluate the model
    print("Traditional Decision Tree Accuracy:", accuracy_traditional)
    print("Traditional Decision Tree Precision:", precision_traditional)
    print("Traditional Decision Tree Recall:", recall_traditional)
    print("Traditional Decision Tree F1-score:", f1_traditional)
    print("Traditional Decision Tree Confusion Matrix:\n", cm_traditional)


    # ====================================================================================================
    # ============================================== Visualize metrics using Bar Chart
    # ====================================================================================================
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score']
    with_fig_scores = [accuracy_custom, precision_custom, recall_custom, f1_custom]
    without_fig_scores = [accuracy_traditional, precision_traditional, recall_traditional, f1_traditional]

    x = np.arange(len(metrics))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    bars1 = ax.bar(x - width / 2, with_fig_scores, width, label='w/ FIG', color='blue')
    bars2 = ax.bar(x + width / 2, without_fig_scores, width, label='w/o FIG', color='orange')

    # Add some text for labels, title, and custom x-axis tick labels, etc.
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Scores')
    ax.set_title('Comparison of Metrics: With FIG vs Without FIG')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()


    # Add values on top of the bars
    def add_labels(bars):
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), ha='center', va='bottom')


    add_labels(bars1)
    add_labels(bars2)

    plt.show()

    # ====================================================================================================
    # ===================================================== Visualize Confusion Matrices using Heatmaps
    # =====================================================================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.heatmap(cm_custom, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title('Confusion Matrix: With FIG')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')

    sns.heatmap(cm_traditional, annot=True, fmt='d', cmap='Oranges', ax=axes[1])
    axes[1].set_title('Confusion Matrix: Without FIG')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')

    plt.tight_layout()
    plt.show()


    # =======================================================================================
    # ============================================== Fairness Evaluation
    # =======================================================================================
    selected_index = X.columns.get_loc("sex")
    selected_feature = X_test[:, selected_index]

    fair_dp_diff, fair_tpr_diff, fair_fpr_diff = calculate_fairness_metrics(y_test, y_pred_custom, selected_feature)
    traditional_dp_diff, traditional_tpr_diff, traditional_fpr_diff = calculate_fairness_metrics(y_test, y_pred,
                                                                                                 selected_feature)

    # Data for visualization
    metrics = ['Demographic Parity', 'TPR Difference', 'FPR Difference']
    fair_metrics = [fair_dp_diff, fair_tpr_diff, fair_fpr_diff]
    traditional_metrics = [traditional_dp_diff, traditional_tpr_diff, traditional_fpr_diff]

    x = np.arange(len(metrics))
    width = 0.35  # width of the bars

    # Plotting
    fig, ax = plt.subplots()
    ax.bar(x - width / 2, fair_metrics, width, label='w/ FIG', color='blue')
    ax.bar(x + width / 2, traditional_metrics, width, label='w/o FIG', color='orange')

    # Labels and titles
    ax.set_xlabel('Fairness Metrics')
    ax.set_ylabel('Difference')
    ax.set_title('Fairness Metrics Comparison: With vs Without FIG')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()

    # Show the plot
    plt.show()


    # ====================================================================================================
    # ===================================================== Discrimination Analysis
    # =====================================================================================================
    selected_index = X.columns.get_loc("sex")
    selected_feature = X_test[:, selected_index]

    # Calculate discrimination scores for both models
    discrimination_score_custom = calculate_discrimination_score(y_test, y_pred_custom, selected_feature)
    discrimination_score_traditional = calculate_discrimination_score(y_test, y_pred, selected_feature)

    print("Discrimination Score for With FIG:", discrimination_score_custom)
    print("Discrimination Score for Without FIG:", discrimination_score_traditional)

    # Visualization
    models = ['With FIG', 'Without FIG']
    discrimination_scores = [discrimination_score_custom, discrimination_score_traditional]

    plt.bar(models, discrimination_scores, color=['blue', 'orange'])
    plt.xlabel("Model")
    plt.ylabel("Discrimination Score")
    plt.title("Discrimination Score Comparison: With vs Without FIG")
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')  # Add a line at y=0 for reference
    plt.show()
