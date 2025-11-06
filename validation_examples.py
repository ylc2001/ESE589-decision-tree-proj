"""
Small illustrating examples for validation of Decision Tree implementation.

This module contains simple toy datasets and tests to validate the correctness
of the Decision Tree implementation with different splitting criteria.
"""

import numpy as np
from decision_tree import DecisionTreeClassifier


def simple_dataset_1():
    """
    Simple binary classification dataset.
    Classic example: Play Tennis
    
    Features: [Outlook, Temperature, Humidity, Windy]
    Encoded as numerical values for simplicity
    """
    # Outlook: 0=Sunny, 1=Overcast, 2=Rainy
    # Temperature: 0=Hot, 1=Mild, 2=Cool
    # Humidity: 0=High, 1=Normal
    # Windy: 0=False, 1=True
    # Play: 0=No, 1=Yes
    
    X = np.array([
        [0, 0, 0, 0],  # Sunny, Hot, High, False -> No
        [0, 0, 0, 1],  # Sunny, Hot, High, True -> No
        [1, 0, 0, 0],  # Overcast, Hot, High, False -> Yes
        [2, 1, 0, 0],  # Rainy, Mild, High, False -> Yes
        [2, 2, 1, 0],  # Rainy, Cool, Normal, False -> Yes
        [2, 2, 1, 1],  # Rainy, Cool, Normal, True -> No
        [1, 2, 1, 1],  # Overcast, Cool, Normal, True -> Yes
        [0, 1, 0, 0],  # Sunny, Mild, High, False -> No
        [0, 2, 1, 0],  # Sunny, Cool, Normal, False -> Yes
        [2, 1, 1, 0],  # Rainy, Mild, Normal, False -> Yes
        [0, 1, 1, 1],  # Sunny, Mild, Normal, True -> Yes
        [1, 1, 0, 1],  # Overcast, Mild, High, True -> Yes
        [1, 0, 1, 0],  # Overcast, Hot, Normal, False -> Yes
        [2, 1, 0, 1],  # Rainy, Mild, High, True -> No
    ])
    
    y = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0])
    
    return X, y


def simple_dataset_2():
    """
    Simple linearly separable dataset.
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


def simple_dataset_3():
    """
    XOR-like dataset (non-linearly separable).
    """
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
        [0.1, 0.1],
        [0.1, 0.9],
        [0.9, 0.1],
        [0.9, 0.9],
    ])
    y = np.array([0, 1, 1, 0, 0, 1, 1, 0])
    
    return X, y


def test_criterion(criterion_name, X_train, y_train, X_test, y_test):
    """
    Test a decision tree with a specific criterion.
    """
    print(f"\n{'='*60}")
    print(f"Testing with criterion: {criterion_name}")
    print(f"{'='*60}")
    
    clf = DecisionTreeClassifier(criterion=criterion_name, max_depth=5)
    clf.fit(X_train, y_train)
    
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    
    print(f"Training accuracy: {train_score:.4f}")
    print(f"Test accuracy: {test_score:.4f}")
    
    # Show some predictions
    predictions = clf.predict(X_test)
    print(f"\nPredictions on test set: {predictions}")
    print(f"True labels:             {y_test}")
    
    return train_score, test_score


def run_validation_tests():
    """
    Run validation tests on small datasets.
    """
    print("\n" + "="*60)
    print("VALIDATION TEST 1: Play Tennis Dataset")
    print("="*60)
    
    X, y = simple_dataset_1()
    
    # Use first 10 samples for training, last 4 for testing
    X_train, X_test = X[:10], X[10:]
    y_train, y_test = y[:10], y[10:]
    
    criteria = ['information_gain', 'gain_ratio', 'gini_index']
    results_1 = {}
    
    for criterion in criteria:
        train_acc, test_acc = test_criterion(criterion, X_train, y_train, X_test, y_test)
        results_1[criterion] = (train_acc, test_acc)
    
    print("\n" + "="*60)
    print("VALIDATION TEST 2: Linearly Separable Dataset")
    print("="*60)
    
    X, y = simple_dataset_2()
    
    # Use first 4 samples for training, last 2 for testing
    X_train, X_test = X[:4], X[4:]
    y_train, y_test = y[:4], y[4:]
    
    results_2 = {}
    
    for criterion in criteria:
        train_acc, test_acc = test_criterion(criterion, X_train, y_train, X_test, y_test)
        results_2[criterion] = (train_acc, test_acc)
    
    print("\n" + "="*60)
    print("VALIDATION TEST 3: XOR-like Dataset")
    print("="*60)
    
    X, y = simple_dataset_3()
    
    # Use first 4 samples for training, last 4 for testing
    X_train, X_test = X[:4], X[4:]
    y_train, y_test = y[:4], y[4:]
    
    results_3 = {}
    
    for criterion in criteria:
        train_acc, test_acc = test_criterion(criterion, X_train, y_train, X_test, y_test)
        results_3[criterion] = (train_acc, test_acc)
    
    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    print("\nDataset 1 (Play Tennis):")
    for criterion, (train_acc, test_acc) in results_1.items():
        print(f"  {criterion:20s}: Train={train_acc:.4f}, Test={test_acc:.4f}")
    
    print("\nDataset 2 (Linearly Separable):")
    for criterion, (train_acc, test_acc) in results_2.items():
        print(f"  {criterion:20s}: Train={train_acc:.4f}, Test={test_acc:.4f}")
    
    print("\nDataset 3 (XOR-like):")
    for criterion, (train_acc, test_acc) in results_3.items():
        print(f"  {criterion:20s}: Train={train_acc:.4f}, Test={test_acc:.4f}")
    
    print("\nValidation tests completed successfully!")


if __name__ == '__main__':
    run_validation_tests()
