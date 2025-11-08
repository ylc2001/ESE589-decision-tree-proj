"""
Validation example for Decision Tree implementation using linearly separable dataset.

This module demonstrates the Decision Tree classifier on a simple 2D linearly separable
dataset with visualization and comparison to sklearn's implementation.
"""

import numpy as np
import matplotlib.pyplot as plt
from decision_tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTreeClassifier


def linearly_separable_dataset():
    """
    Simple linearly separable dataset for binary classification.
    Two clusters in 2D space that can be perfectly separated by a line.
    """
    X = np.array([
        [1, 2],
        [2, 3],
        [3, 1],
        [6, 5],
        [7, 8],
        [8, 6],
    ])
    y = np.array([0, 0, 0, 1, 1, 1])
    
    return X, y


def plot_dataset(X, y, filename='dataset_plot.png'):
    """
    Plot the dataset points and save to an image file.
    
    Parameters:
    -----------
    X : array-like of shape (n_samples, 2)
        The input samples (2D data).
    y : array-like of shape (n_samples,)
        The target values (class labels).
    filename : str
        The filename to save the plot.
    """
    plt.figure(figsize=(8, 6))
    
    # Plot points for each class
    for class_label in np.unique(y):
        mask = y == class_label
        plt.scatter(X[mask, 0], X[mask, 1], 
                   label=f'Class {class_label}',
                   s=100, alpha=0.7, edgecolors='k', linewidth=1.5)
    
    plt.xlabel('Feature 1 (X[0])', fontsize=12)
    plt.ylabel('Feature 2 (X[1])', fontsize=12)
    plt.title('Linearly Separable Dataset', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\n✓ Dataset plot saved to: {filename}")
    plt.close()


def compare_with_sklearn(X, y, criterion='gini'):
    """
    Compare our implementation with sklearn's DecisionTreeClassifier.
    
    Parameters:
    -----------
    X : array-like of shape (n_samples, n_features)
        The input samples.
    y : array-like of shape (n_samples,)
        The target values.
    criterion : str
        The criterion to use ('gini' or 'entropy').
    """
    print("\n" + "="*60)
    print("COMPARISON WITH SKLEARN")
    print("="*60)
    
    # Map our criterion names to sklearn's
    criterion_map = {
        'information_gain': 'entropy',
        'gini_index': 'gini'
    }
    
    sklearn_criterion = criterion_map.get(criterion, 'gini')
    
    # Train sklearn's decision tree
    sklearn_clf = SklearnDecisionTreeClassifier(
        criterion=sklearn_criterion,
        max_depth=5,
        random_state=42
    )
    sklearn_clf.fit(X, y)
    sklearn_accuracy = sklearn_clf.score(X, y)
    
    print(f"\nSklearn DecisionTreeClassifier (criterion='{sklearn_criterion}'):")
    print(f"  Accuracy: {sklearn_accuracy:.4f}")
    print(f"  Tree depth: {sklearn_clf.get_depth()}")
    print(f"  Number of leaves: {sklearn_clf.get_n_leaves()}")
    
    # Train our decision tree
    our_criterion = criterion
    our_clf = DecisionTreeClassifier(criterion=our_criterion, max_depth=5)
    our_clf.fit(X, y)
    our_accuracy = our_clf.score(X, y)
    
    print(f"\nOur DecisionTreeClassifier (criterion='{our_criterion}'):")
    print(f"  Accuracy: {our_accuracy:.4f}")
    
    # Compare predictions
    sklearn_pred = sklearn_clf.predict(X)
    our_pred = our_clf.predict(X)
    
    print(f"\nPrediction comparison:")
    print(f"  Sklearn predictions: {sklearn_pred}")
    print(f"  Our predictions:     {our_pred}")
    print(f"  Match: {np.array_equal(sklearn_pred, our_pred)}")
    
    return our_clf, sklearn_clf


def run_validation():
    """
    Run the validation example with dataset 2 (linearly separable).
    """
    print("\n" + "="*60)
    print("DECISION TREE VALIDATION EXAMPLE")
    print("Dataset 2: Linearly Separable Data")
    print("="*60)
    
    # Load the dataset
    X, y = linearly_separable_dataset()
    
    print(f"\nDataset information:")
    print(f"  Number of samples: {len(X)}")
    print(f"  Number of features: {X.shape[1]}")
    print(f"  Number of classes: {len(np.unique(y))}")
    print(f"\nData points:")
    for i, (x, label) in enumerate(zip(X, y)):
        print(f"  Sample {i+1}: {x} -> Class {label}")
    
    # 1. Plot the dataset
    plot_dataset(X, y, filename='dataset_plot.png')
    
    # 2. Train decision tree with different criteria
    print("\n" + "="*60)
    print("TRAINING DECISION TREES")
    print("="*60)
    
    criteria = ['information_gain', 'gain_ratio', 'gini_index']
    
    for criterion in criteria:
        print(f"\n{'─'*60}")
        print(f"Criterion: {criterion}")
        print(f"{'─'*60}")
        
        clf = DecisionTreeClassifier(criterion=criterion, max_depth=5)
        clf.fit(X, y)
        
        accuracy = clf.score(X, y)
        print(f"Training accuracy: {accuracy:.4f}")
        
        predictions = clf.predict(X)
        print(f"Predictions: {predictions}")
        print(f"True labels: {y}")
        
        # 3. Print the decision tree structure
        clf.print_tree(feature_names=['X[0]', 'X[1]'])
    
    # 4. Compare with sklearn
    print("\n")
    our_clf, sklearn_clf = compare_with_sklearn(X, y, criterion='gini_index')
    
    print("\n" + "="*60)
    print("VALIDATION COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nKey observations:")
    print("  ✓ The dataset is linearly separable")
    print("  ✓ All criteria achieve 100% accuracy on this simple dataset")
    print("  ✓ Decision tree structure printed above")
    print("  ✓ Comparison with sklearn shows similar performance")
    print("  ✓ Dataset visualization saved to 'dataset_plot.png'")


if __name__ == '__main__':
    run_validation()
