"""
Decision Tree Induction Algorithm for Classification

This module implements the Decision Tree algorithm with three different
splitting criteria: Information Gain, Gain Ratio, and Gini Index.
"""

import numpy as np
from collections import Counter
from typing import Optional, Union, Literal


class Node:
    """
    Represents a node in the decision tree.
    """
    def __init__(
        self,
        feature: Optional[int] = None,
        threshold: Optional[float] = None,
        left: Optional['Node'] = None,
        right: Optional['Node'] = None,
        value: Optional[int] = None,
        is_leaf: bool = False
    ):
        self.feature = feature  # Index of feature to split on
        self.threshold = threshold  # Threshold value for the split
        self.left = left  # Left child node
        self.right = right  # Right child node
        self.value = value  # Class label if leaf node
        self.is_leaf = is_leaf  # Whether this is a leaf node


class DecisionTreeClassifier:
    """
    Decision Tree Classifier with multiple splitting criteria.
    
    Parameters:
    -----------
    criterion : str, default='information_gain'
        The function to measure the quality of a split.
        Supported criteria are 'information_gain', 'gain_ratio', and 'gini_index'.
    max_depth : int, default=None
        The maximum depth of the tree. If None, nodes are expanded until
        all leaves are pure or contain less than min_samples_split samples.
    min_samples_split : int, default=2
        The minimum number of samples required to split an internal node.
    min_samples_leaf : int, default=1
        The minimum number of samples required to be at a leaf node.
    """
    
    def __init__(
        self,
        criterion: Literal['information_gain', 'gain_ratio', 'gini_index'] = 'information_gain',
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1
    ):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.root = None
        self.n_classes_ = None
        self.n_features_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Build a decision tree classifier from the training set (X, y).
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values (class labels). Must be non-negative integers.
        
        Returns:
        --------
        self : DecisionTreeClassifier
            Fitted estimator.
        """
        X = np.array(X)
        y = np.array(y)
        
        # Validate input labels
        if not np.issubdtype(y.dtype, np.integer):
            raise ValueError("Target labels must be integers")
        if np.any(y < 0):
            raise ValueError("Target labels must be non-negative")
        
        self.n_classes_ = len(np.unique(y))
        self.n_features_ = X.shape[1]
        
        self.root = self._build_tree(X, y, depth=0)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples in X.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        
        Returns:
        --------
        y : array-like of shape (n_samples,)
            The predicted classes.
        """
        X = np.array(X)
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        """
        Recursively build the decision tree.
        """
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        
        # Stopping criteria
        if (depth == self.max_depth or 
            n_labels == 1 or 
            n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value, is_leaf=True)
        
        # Find the best split
        best_feature, best_threshold = self._best_split(X, y)
        
        if best_feature is None:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value, is_leaf=True)
        
        # Split the data
        left_idxs = X[:, best_feature] <= best_threshold
        right_idxs = ~left_idxs
        
        # Check minimum samples per leaf
        if (np.sum(left_idxs) < self.min_samples_leaf or 
            np.sum(right_idxs) < self.min_samples_leaf):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value, is_leaf=True)
        
        # Recursively build left and right subtrees
        left = self._build_tree(X[left_idxs], y[left_idxs], depth + 1)
        right = self._build_tree(X[right_idxs], y[right_idxs], depth + 1)
        
        return Node(feature=best_feature, threshold=best_threshold, 
                   left=left, right=right)
    
    def _best_split(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """
        Find the best split for a node.
        """
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        for feature_idx in range(X.shape[1]):
            feature_values = X[:, feature_idx]
            thresholds = np.unique(feature_values)
            
            for threshold in thresholds:
                gain = self._calculate_gain(X, y, feature_idx, threshold)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def _calculate_gain(self, X: np.ndarray, y: np.ndarray, 
                       feature_idx: int, threshold: float) -> float:
        """
        Calculate the information gain of a split.
        """
        if self.criterion == 'information_gain':
            return self._information_gain(X, y, feature_idx, threshold)
        elif self.criterion == 'gain_ratio':
            return self._gain_ratio(X, y, feature_idx, threshold)
        elif self.criterion == 'gini_index':
            return self._gini_gain(X, y, feature_idx, threshold)
        else:
            raise ValueError(f"Unknown criterion: {self.criterion}")
    
    def _information_gain(self, X: np.ndarray, y: np.ndarray, 
                         feature_idx: int, threshold: float) -> float:
        """
        Calculate information gain using entropy.
        
        IG(S, A) = Entropy(S) - Sum((|Sv| / |S|) * Entropy(Sv))
        """
        # Parent entropy
        parent_entropy = self._entropy(y)
        
        # Split the data
        left_idxs = X[:, feature_idx] <= threshold
        right_idxs = ~left_idxs
        
        if np.sum(left_idxs) == 0 or np.sum(right_idxs) == 0:
            return 0
        
        # Calculate weighted average of children entropy
        n = len(y)
        n_left, n_right = np.sum(left_idxs), np.sum(right_idxs)
        e_left, e_right = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_left / n) * e_left + (n_right / n) * e_right
        
        # Information gain
        return parent_entropy - child_entropy
    
    def _gain_ratio(self, X: np.ndarray, y: np.ndarray, 
                   feature_idx: int, threshold: float) -> float:
        """
        Calculate gain ratio to avoid bias towards features with many values.
        
        GainRatio(S, A) = IG(S, A) / SplitInfo(S, A)
        SplitInfo(S, A) = -Sum((|Sv| / |S|) * log2(|Sv| / |S|))
        """
        # Calculate information gain
        info_gain = self._information_gain(X, y, feature_idx, threshold)
        
        if info_gain == 0:
            return 0
        
        # Calculate split information
        left_idxs = X[:, feature_idx] <= threshold
        right_idxs = ~left_idxs
        
        n = len(y)
        n_left, n_right = np.sum(left_idxs), np.sum(right_idxs)
        
        if n_left == 0 or n_right == 0:
            return 0
        
        split_info = 0
        for n_subset in [n_left, n_right]:
            if n_subset > 0:
                p = n_subset / n
                split_info -= p * np.log2(p)
        
        if split_info == 0:
            return 0
        
        # Gain ratio
        return info_gain / split_info
    
    def _gini_gain(self, X: np.ndarray, y: np.ndarray, 
                  feature_idx: int, threshold: float) -> float:
        """
        Calculate Gini gain (reduction in Gini impurity).
        
        Gini(S) = 1 - Sum(pi^2)
        GiniGain = Gini(parent) - weighted_average(Gini(children))
        """
        # Parent Gini impurity
        parent_gini = self._gini_impurity(y)
        
        # Split the data
        left_idxs = X[:, feature_idx] <= threshold
        right_idxs = ~left_idxs
        
        if np.sum(left_idxs) == 0 or np.sum(right_idxs) == 0:
            return 0
        
        # Calculate weighted average of children Gini impurity
        n = len(y)
        n_left, n_right = np.sum(left_idxs), np.sum(right_idxs)
        gini_left = self._gini_impurity(y[left_idxs])
        gini_right = self._gini_impurity(y[right_idxs])
        child_gini = (n_left / n) * gini_left + (n_right / n) * gini_right
        
        # Gini gain
        return parent_gini - child_gini
    
    def _entropy(self, y: np.ndarray) -> float:
        """
        Calculate entropy of a label array.
        
        Entropy(S) = -Sum(pi * log2(pi))
        """
        # Use minlength to ensure consistent output size
        proportions = np.bincount(y, minlength=self.n_classes_) / len(y)
        entropy = 0
        for p in proportions:
            if p > 0:
                entropy -= p * np.log2(p)
        return entropy
    
    def _gini_impurity(self, y: np.ndarray) -> float:
        """
        Calculate Gini impurity of a label array.
        
        Gini(S) = 1 - Sum(pi^2)
        """
        # Use minlength to ensure consistent output size
        proportions = np.bincount(y, minlength=self.n_classes_) / len(y)
        return 1 - np.sum(proportions ** 2)
    
    def _most_common_label(self, y: np.ndarray) -> int:
        """
        Return the most common label in y.
        """
        counter = Counter(y)
        return counter.most_common(1)[0][0]
    
    def _traverse_tree(self, x: np.ndarray, node: Node) -> int:
        """
        Traverse the tree to make a prediction for a single sample.
        """
        if node.is_leaf:
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return the mean accuracy on the given test data and labels.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True labels for X.
        
        Returns:
        --------
        score : float
            Mean accuracy of self.predict(X) wrt. y.
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
    
    def print_tree(self, feature_names=None):
        """
        Print the decision tree structure in a human-readable format.
        
        Parameters:
        -----------
        feature_names : list of str, optional
            Names of the features. If None, features are referred to by their index.
        """
        if self.root is None:
            print("Tree has not been trained yet.")
            return
        
        def _print_node(node, depth=0, prefix="Root: "):
            indent = "  " * depth
            
            if node.is_leaf:
                print(f"{indent}{prefix}Predict class {node.value}")
            else:
                feature_name = (feature_names[node.feature] 
                               if feature_names else f"X[{node.feature}]")
                print(f"{indent}{prefix}if {feature_name} <= {node.threshold:.4f}:")
                _print_node(node.left, depth + 1, "├─ True: ")
                print(f"{indent}  else:")
                _print_node(node.right, depth + 1, "└─ False: ")
        
        print("\n" + "="*60)
        print("Decision Tree Structure")
        print("="*60)
        _print_node(self.root)
        print("="*60 + "\n")
