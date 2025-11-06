"""
Simple example demonstrating how to use the Decision Tree classifier.
"""

from decision_tree import DecisionTreeClassifier
import numpy as np


def main():
    print("="*60)
    print("Decision Tree Classifier - Simple Example")
    print("="*60)
    
    # Create a simple dataset
    # Features: [Feature1, Feature2]
    # Classes: 0 or 1
    X_train = np.array([
        [1, 2],
        [2, 3],
        [3, 1],
        [6, 5],
        [7, 8],
        [8, 6],
    ])
    y_train = np.array([0, 0, 0, 1, 1, 1])
    
    print("\nTraining Data:")
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("\nX_train:")
    print(X_train)
    print("\ny_train:", y_train)
    
    # Test data
    X_test = np.array([
        [2, 2],
        [7, 7],
        [1, 1],
        [9, 9],
    ])
    y_test = np.array([0, 1, 0, 1])
    
    print("\n" + "="*60)
    print("Training classifiers with different criteria")
    print("="*60)
    
    criteria = ['information_gain', 'gain_ratio', 'gini_index']
    
    for criterion in criteria:
        print(f"\n{criterion.upper()}")
        print("-" * 40)
        
        # Create and train classifier
        clf = DecisionTreeClassifier(criterion=criterion, max_depth=5)
        clf.fit(X_train, y_train)
        
        # Make predictions
        y_pred = clf.predict(X_test)
        
        # Calculate accuracy
        train_acc = clf.score(X_train, y_train)
        test_acc = clf.score(X_test, y_test)
        
        print(f"Training accuracy: {train_acc:.2%}")
        print(f"Test accuracy: {test_acc:.2%}")
        print(f"Predictions: {y_pred}")
        print(f"True labels: {y_test}")
    
    print("\n" + "="*60)
    print("Example completed!")
    print("="*60)


if __name__ == '__main__':
    main()
