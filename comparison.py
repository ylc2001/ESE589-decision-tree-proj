"""
Comparison of the three splitting criteria on various datasets.

This script demonstrates the differences between Information Gain,
Gain Ratio, and Gini Index on different types of datasets.
"""

import numpy as np
from decision_tree import DecisionTreeClassifier
from sklearn.datasets import make_classification, make_moons, make_circles
from sklearn.model_selection import train_test_split
import time


def evaluate_on_dataset(X, y, dataset_name):
    """
    Evaluate all three criteria on a given dataset.
    """
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*60}")
    print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    criteria = ['information_gain', 'gain_ratio', 'gini_index']
    results = []
    
    print(f"\n{'Criterion':<20} {'Train Acc':<12} {'Test Acc':<12} {'Time (s)':<10}")
    print("-" * 60)
    
    for criterion in criteria:
        clf = DecisionTreeClassifier(criterion=criterion, max_depth=10)
        
        start = time.time()
        clf.fit(X_train, y_train)
        train_time = time.time() - start
        
        train_acc = clf.score(X_train, y_train)
        test_acc = clf.score(X_test, y_test)
        
        results.append({
            'criterion': criterion,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'time': train_time
        })
        
        print(f"{criterion:<20} {train_acc:<12.4f} {test_acc:<12.4f} {train_time:<10.4f}")
    
    # Find best
    best = max(results, key=lambda x: x['test_acc'])
    print(f"\nBest: {best['criterion']} with {best['test_acc']:.4f} test accuracy")
    
    return results


def main():
    print("="*60)
    print("Splitting Criteria Comparison")
    print("="*60)
    
    # Dataset 1: Linearly separable
    print("\n" + "="*60)
    print("EXPERIMENT 1: Linearly Separable Data")
    print("="*60)
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=1,
        random_state=42
    )
    evaluate_on_dataset(X, y, "Linearly Separable")
    
    # Dataset 2: More complex
    print("\n" + "="*60)
    print("EXPERIMENT 2: Complex Decision Boundary")
    print("="*60)
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=8,
        n_redundant=1,
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=3,
        random_state=42
    )
    evaluate_on_dataset(X, y, "Complex Boundary")
    
    # Dataset 3: Imbalanced
    print("\n" + "="*60)
    print("EXPERIMENT 3: Imbalanced Classes")
    print("="*60)
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_repeated=0,
        n_classes=2,
        weights=[0.9, 0.1],
        random_state=42
    )
    evaluate_on_dataset(X, y, "Imbalanced Classes")
    
    # Dataset 4: Many features
    print("\n" + "="*60)
    print("EXPERIMENT 4: High Dimensional Data")
    print("="*60)
    X, y = make_classification(
        n_samples=1000,
        n_features=50,
        n_informative=20,
        n_redundant=10,
        n_repeated=0,
        n_classes=2,
        random_state=42
    )
    evaluate_on_dataset(X, y, "High Dimensional")
    
    # Dataset 5: Moons (non-linear)
    print("\n" + "="*60)
    print("EXPERIMENT 5: Non-linear Decision Boundary (Moons)")
    print("="*60)
    X, y = make_moons(n_samples=1000, noise=0.3, random_state=42)
    evaluate_on_dataset(X, y, "Moons Dataset")
    
    # Dataset 6: Circles (very non-linear)
    print("\n" + "="*60)
    print("EXPERIMENT 6: Very Non-linear Boundary (Circles)")
    print("="*60)
    X, y = make_circles(n_samples=1000, noise=0.2, factor=0.5, random_state=42)
    evaluate_on_dataset(X, y, "Circles Dataset")
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print("\nKey Observations:")
    print("1. All three criteria perform similarly on most datasets")
    print("2. Gain Ratio is often faster due to early stopping")
    print("3. Gini Index and Information Gain typically have similar performance")
    print("4. The choice of criterion may depend on the specific problem")
    print("\nRecommendations:")
    print("- Use Information Gain for interpretable splits")
    print("- Use Gain Ratio for datasets with features of varying cardinality")
    print("- Use Gini Index for computational efficiency")


if __name__ == '__main__':
    main()
