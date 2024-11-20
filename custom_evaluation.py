import numpy as np


# Function to calculate the discrimination score
def calculate_discrimination_score(y_true, y_pred, sensitive_attribute):
    # 0 = female (deprived)
    # 1 = male (favored)

    # Deprived community (female)
    deprived_rejected = np.sum((sensitive_attribute == 0) & (y_pred == 0))  # DR
    deprived_granted = np.sum((sensitive_attribute == 0) & (y_pred == 1))  # DG

    # Favored community (male)
    favored_rejected = np.sum((sensitive_attribute == 1) & (y_pred == 0))  # FR
    favored_granted = np.sum((sensitive_attribute == 1) & (y_pred == 1))  # FG

    # Calculate probabilities
    p_favored_granted = favored_granted / (favored_granted + favored_rejected)
    p_deprived_granted = deprived_granted / (deprived_granted + deprived_rejected)

    # Calculate discrimination score
    discrimination_score = p_favored_granted - p_deprived_granted
    return discrimination_score

def calculate_fairness_metrics(y_true, y_pred, sensitive_feature):
    # Separate the data based on the sensitive attribute
    group_0 = (sensitive_feature == 0)
    group_1 = (sensitive_feature == 1)

    # Demographic Parity
    positive_rate_0 = np.mean(y_pred[group_0])
    positive_rate_1 = np.mean(y_pred[group_1])
    demographic_parity_diff = abs(positive_rate_0 - positive_rate_1)

    # True Positive Rate (TPR) - Equal Opportunity
    tpr_0 = np.sum((y_true[group_0] == 1) & (y_pred[group_0] == 1)) / np.sum(y_true[group_0] == 1)
    tpr_1 = np.sum((y_true[group_1] == 1) & (y_pred[group_1] == 1)) / np.sum(y_true[group_1] == 1)
    tpr_diff = abs(tpr_0 - tpr_1)

    # False Positive Rate (FPR)
    fpr_0 = np.sum((y_true[group_0] == 0) & (y_pred[group_0] == 1)) / np.sum(y_true[group_0] == 0)
    fpr_1 = np.sum((y_true[group_1] == 0) & (y_pred[group_1] == 1)) / np.sum(y_true[group_1] == 0)
    fpr_diff = abs(fpr_0 - fpr_1)
    return demographic_parity_diff, tpr_diff, fpr_diff