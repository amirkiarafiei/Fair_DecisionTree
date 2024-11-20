import numpy as np


# Compute Entropy and Information Gain
def calculate_entropy(y):
    class_counts = np.bincount(y)
    probabilities = class_counts / len(y)
    return -np.sum([p * np.log2(p) for p in probabilities if p > 0])


def information_gain(y, y_left, y_right):
    entropy_before = calculate_entropy(y)
    entropy_after = (len(y_left) / len(y)) * calculate_entropy(y_left) + (len(y_right) / len(y)) * calculate_entropy(
        y_right)
    return entropy_before - entropy_after


# Compute Discrimination Score
def discrimination_score(y, sensitive_attribute):
    deprived_positive = np.sum((y == 1) & (sensitive_attribute == 0))
    deprived_total = np.sum(sensitive_attribute == 0)
    favored_positive = np.sum((y == 1) & (sensitive_attribute == 1))
    favored_total = np.sum(sensitive_attribute == 1)

    if deprived_total == 0 or favored_total == 0:
        return 0  # Avoid division by zero

    disc_deprived = deprived_positive / deprived_total
    disc_favored = favored_positive / favored_total
    return disc_favored - disc_deprived


# Compute Fair Information Gain (FIG)
def fair_information_gain(X, y, feature_index, protected_indices):
    if len(y) == 0:
        return 0
    # Boolean Masks for the left and right nodes (Trues and Falses)
    left_indices = X[:, feature_index] <= np.median(X[:, feature_index])
    right_indices = ~left_indices

    # boolean indexing
    y_left, y_right = y[left_indices], y[right_indices]

    # example of boolean indexing for myself
    # left_indices = [True, False, True, False, True]
    # y = [1, 2, 3, 4, 5]
    # y[left_indices] = [1, 3, 5]
    #
    # Get the indices of the elements that satisfy the condition using integer indexing
    # left_indices = np.where(X[:, feature_index] <= np.median(X[:, feature_index]))[0]
    # right_indices = np.where(X[:, feature_index] > np.median(X[:, feature_index]))[0]

    # Calculate IG
    ig = information_gain(y, y_left, y_right)

    # If no protected indices are provided, return standard Information Gain
    if not protected_indices:
        return ig

    # Calculate Fairness Gain (FG) if protected indices are provided
    # Using boolean indexing (first bracket uses the trues anf falses list to filter the rows, second bracket
    # select all the rows for the protected attribute)
    # X[left_indices] is a 2D or more array
    sensitive_left = X[left_indices][:, protected_indices]
    sensitive_right = X[right_indices][:, protected_indices]

    disc_before = np.mean([abs(discrimination_score(y, X[:, i])) for i in protected_indices])
    disc_left = np.mean([abs(discrimination_score(y_left, sensitive_left[:, i])) for i in range(len(protected_indices))])
    disc_right = np.mean([abs(discrimination_score(y_right, sensitive_right[:, i])) for i in range(len(protected_indices))])

    fg = disc_before - (len(y_left) / len(y)) * disc_left - (len(y_right) / len(y)) * disc_right
    # print(fg)

    # Calculate FIG
    if fg == 0:
        return ig
    return ig * fg


# Custom Decision Tree Classifier with FIG as Criterion
class CustomDecisionTreeNode:
    def __init__(self, depth=0, max_depth=None):
        self.depth = depth
        self.max_depth = max_depth
        self.feature_index = None
        self.threshold = None
        self.left = None
        self.right = None
        self.is_leaf = False
        self.prediction = None

    def fit(self, X, y, protected_indices):
        if self.depth == self.max_depth or len(np.unique(y)) == 1:
            self.is_leaf = True
            # self.prediction = np.argmax(np.bincount(y))
            if len(y) > 0:
                self.prediction = np.argmax(np.bincount(y))
            else:
                self.prediction = None
            return

        best_fig = -np.inf
        best_feature_index = None
        best_threshold = None

        for feature_index in range(X.shape[1]):
            # if feature_index in protected_indices:
            #     continue  # Skip protected attributes

            threshold = np.median(X[:, feature_index])
            fig = fair_information_gain(X, y, feature_index, protected_indices)

            if fig > best_fig:
                best_fig = fig
                best_feature_index = feature_index
                best_threshold = threshold

        if best_feature_index is None:
            self.is_leaf = True
            self.prediction = np.argmax(np.bincount(y))
            return

        self.feature_index = best_feature_index
        self.threshold = best_threshold

        left_indices = X[:, self.feature_index] <= self.threshold
        right_indices = ~left_indices

        self.left = CustomDecisionTreeNode(depth=self.depth + 1, max_depth=self.max_depth)
        self.right = CustomDecisionTreeNode(depth=self.depth + 1, max_depth=self.max_depth)

        self.left.fit(X[left_indices], y[left_indices], protected_indices)
        self.right.fit(X[right_indices], y[right_indices], protected_indices)

    def predict(self, x):
        if self.is_leaf:
            return self.prediction
        if x[self.feature_index] <= self.threshold:
            return self.left.predict(x)
        else:
            return self.right.predict(x)

class CustomFairDecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.root = CustomDecisionTreeNode(max_depth=max_depth)

    def fit(self, X, y, protected_indices):
        self.root.fit(X, y, protected_indices)

    def predict(self, X):
        return np.array([self.root.predict(x) for x in X])




