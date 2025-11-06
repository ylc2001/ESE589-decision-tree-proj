"""
Benchmark experiments with synthetic data (since UCI dataset cannot be downloaded).

This script creates a large synthetic dataset similar to the Census Income dataset
and evaluates the Decision Tree classifier with different splitting criteria.
"""

import numpy as np
import time
from decision_tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


def create_synthetic_census_data(n_samples=10000, n_features=14, random_state=42):
    """
    Create a synthetic dataset similar to the Census Income dataset.
    
    The real Census Income dataset has:
    - ~48,000 samples
    - 14 features (mix of categorical and numerical)
    - Binary classification (<=50K, >50K)
    """
    print(f"Creating synthetic dataset with {n_samples} samples and {n_features} features...")
    
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=10,
        n_redundant=2,
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=3,
        weights=[0.75, 0.25],  # Imbalanced like real Census data
        flip_y=0.01,
        random_state=random_state
    )
    
    # Scale features to make them look more realistic
    X = np.abs(X * 10).astype(int)
    
    print(f"Dataset created: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class distribution: {np.bincount(y)}")
    
    return X, y


def evaluate_criterion(criterion_name, X_train, y_train, X_test, y_test, 
                       max_depth=10):
    """
    Evaluate a decision tree with a specific criterion.
    """
    print(f"\n{'='*60}")
    print(f"Evaluating: {criterion_name}")
    print(f"{'='*60}")
    
    # Train the model
    clf = DecisionTreeClassifier(
        criterion=criterion_name,
        max_depth=max_depth,
        min_samples_split=20,
        min_samples_leaf=10
    )
    
    print("Training model...")
    start_time = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    print(f"Training time: {train_time:.2f} seconds")
    
    # Evaluate on training set
    print("Evaluating on training set...")
    start_time = time.time()
    train_score = clf.score(X_train, y_train)
    train_pred_time = time.time() - start_time
    
    # Evaluate on test set
    print("Evaluating on test set...")
    start_time = time.time()
    test_score = clf.score(X_test, y_test)
    test_pred_time = time.time() - start_time
    
    print(f"\nResults:")
    print(f"  Training accuracy:   {train_score:.4f}")
    print(f"  Test accuracy:       {test_score:.4f}")
    print(f"  Training time:       {train_time:.2f} seconds")
    print(f"  Train predict time:  {train_pred_time:.4f} seconds")
    print(f"  Test predict time:   {test_pred_time:.4f} seconds")
    
    return {
        'criterion': criterion_name,
        'train_accuracy': train_score,
        'test_accuracy': test_score,
        'train_time': train_time,
        'train_pred_time': train_pred_time,
        'test_pred_time': test_pred_time
    }


def run_benchmark(n_samples=10000, max_depth=10, test_size=0.2):
    """
    Run benchmark experiments on synthetic Census-like data.
    """
    print("="*60)
    print("SYNTHETIC CENSUS-LIKE DATA BENCHMARK")
    print("="*60)
    
    # Create synthetic data
    X, y = create_synthetic_census_data(n_samples=n_samples)
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    print(f"\nTraining set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")
    
    # Test all three criteria
    criteria = ['information_gain', 'gain_ratio', 'gini_index']
    results = []
    
    print(f"\nConfiguration:")
    print(f"  Max depth: {max_depth}")
    
    for criterion in criteria:
        result = evaluate_criterion(
            criterion, X_train, y_train, X_test, y_test,
            max_depth=max_depth
        )
        results.append(result)
    
    # Summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    print("\nComparison of splitting criteria:")
    print(f"{'Criterion':<20} {'Train Acc':<12} {'Test Acc':<12} {'Train Time':<12}")
    print("-" * 60)
    
    for result in results:
        print(f"{result['criterion']:<20} "
              f"{result['train_accuracy']:<12.4f} "
              f"{result['test_accuracy']:<12.4f} "
              f"{result['train_time']:<12.2f}")
    
    # Find best criterion
    best_test = max(results, key=lambda x: x['test_accuracy'])
    print(f"\nBest test accuracy: {best_test['criterion']} ({best_test['test_accuracy']:.4f})")
    
    return results


def run_detailed_experiments():
    """
    Run detailed experiments with different configurations.
    """
    print("\n" + "="*60)
    print("DETAILED EXPERIMENTS")
    print("="*60)
    
    # Experiment 1: Effect of max_depth
    print("\n" + "="*60)
    print("Experiment 1: Effect of max_depth")
    print("="*60)
    
    X, y = create_synthetic_census_data(n_samples=5000)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    depths = [3, 5, 10, 15]
    criterion = 'information_gain'
    
    print(f"\nUsing {criterion}")
    print(f"{'Max Depth':<12} {'Train Acc':<12} {'Test Acc':<12} {'Train Time':<12}")
    print("-" * 60)
    
    for depth in depths:
        result = evaluate_criterion(
            criterion, X_train, y_train, X_test, y_test,
            max_depth=depth
        )
        print(f"{depth:<12} "
              f"{result['train_accuracy']:<12.4f} "
              f"{result['test_accuracy']:<12.4f} "
              f"{result['train_time']:<12.2f}")
    
    # Experiment 2: Different dataset sizes
    print("\n" + "="*60)
    print("Experiment 2: Effect of dataset size")
    print("="*60)
    
    dataset_sizes = [1000, 2500, 5000, 10000]
    max_depth = 10
    
    print(f"\nUsing {criterion} with max_depth={max_depth}")
    print(f"{'Dataset Size':<12} {'Train Acc':<12} {'Test Acc':<12} {'Train Time':<12}")
    print("-" * 60)
    
    for size in dataset_sizes:
        X, y = create_synthetic_census_data(n_samples=size)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        result = evaluate_criterion(
            criterion, X_train, y_train, X_test, y_test,
            max_depth=max_depth
        )
        print(f"{size:<12} "
              f"{result['train_accuracy']:<12.4f} "
              f"{result['test_accuracy']:<12.4f} "
              f"{result['train_time']:<12.2f}")
    
    # Experiment 3: Compare all criteria on larger dataset
    print("\n" + "="*60)
    print("Experiment 3: Criteria comparison on larger dataset")
    print("="*60)
    
    X, y = create_synthetic_census_data(n_samples=20000)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    criteria = ['information_gain', 'gain_ratio', 'gini_index']
    
    print(f"\nDataset: 20,000 samples, max_depth=10")
    print(f"{'Criterion':<20} {'Train Acc':<12} {'Test Acc':<12} {'Train Time':<12}")
    print("-" * 60)
    
    for criterion in criteria:
        result = evaluate_criterion(
            criterion, X_train, y_train, X_test, y_test,
            max_depth=10
        )
        print(f"{result['criterion']:<20} "
              f"{result['train_accuracy']:<12.4f} "
              f"{result['test_accuracy']:<12.4f} "
              f"{result['train_time']:<12.2f}")


if __name__ == '__main__':
    # Run basic benchmark
    results = run_benchmark(n_samples=10000, max_depth=10)
    
    # Run detailed experiments
    run_detailed_experiments()
    
    print("\n" + "="*60)
    print("Benchmark completed!")
    print("="*60)
