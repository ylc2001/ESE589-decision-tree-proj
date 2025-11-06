"""
Benchmark experiments on the Census Income dataset.

This script downloads the Adult Census Income dataset from UCI ML Repository
and evaluates the Decision Tree classifier with different splitting criteria.
"""

import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from decision_tree import DecisionTreeClassifier
import requests
import os


def download_census_data():
    """
    Download the Census Income dataset from UCI ML Repository.
    """
    # URLs for the dataset
    train_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    test_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
    
    # Column names
    columns = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num',
        'marital-status', 'occupation', 'relationship', 'race', 'sex',
        'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
    ]
    
    print("Downloading Census Income dataset...")
    
    # Download training data
    if not os.path.exists('adult.data'):
        response = requests.get(train_url)
        with open('adult.data', 'wb') as f:
            f.write(response.content)
        print("Training data downloaded.")
    else:
        print("Training data already exists.")
    
    # Download test data
    if not os.path.exists('adult.test'):
        response = requests.get(test_url)
        with open('adult.test', 'wb') as f:
            f.write(response.content)
        print("Test data downloaded.")
    else:
        print("Test data already exists.")
    
    # Load the data
    train_df = pd.read_csv('adult.data', names=columns, skipinitialspace=True, na_values='?')
    test_df = pd.read_csv('adult.test', names=columns, skipinitialspace=True, 
                         skiprows=1, na_values='?')
    
    # Clean the income column in test data (remove trailing '.')
    test_df['income'] = test_df['income'].str.rstrip('.')
    
    return train_df, test_df


def preprocess_data(train_df, test_df):
    """
    Preprocess the Census Income dataset.
    """
    print("\nPreprocessing data...")
    
    # Combine train and test for consistent encoding
    combined_df = pd.concat([train_df, test_df], axis=0)
    
    # Drop rows with missing values
    combined_df = combined_df.dropna()
    
    # Separate features and target
    X_combined = combined_df.drop('income', axis=1)
    y_combined = combined_df['income']
    
    # Encode target variable
    le_target = LabelEncoder()
    y_combined = le_target.fit_transform(y_combined)
    
    # Encode categorical features
    categorical_cols = X_combined.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        le = LabelEncoder()
        X_combined[col] = le.fit_transform(X_combined[col])
    
    # Split back into train and test
    n_train = len(train_df.dropna())
    X_train = X_combined.iloc[:n_train].values
    y_train = y_combined[:n_train]
    X_test = X_combined.iloc[n_train:].values
    y_test = y_combined[n_train:]
    
    print(f"Training set size: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Test set size: {X_test.shape[0]} samples")
    print(f"Class distribution in training: {np.bincount(y_train)}")
    
    return X_train, X_test, y_train, y_test


def evaluate_criterion(criterion_name, X_train, y_train, X_test, y_test, 
                       max_depth=10, sample_size=None):
    """
    Evaluate a decision tree with a specific criterion.
    """
    print(f"\n{'='*60}")
    print(f"Evaluating: {criterion_name}")
    print(f"{'='*60}")
    
    # Sample data if specified (for faster training)
    if sample_size and sample_size < len(X_train):
        indices = np.random.choice(len(X_train), sample_size, replace=False)
        X_train_sample = X_train[indices]
        y_train_sample = y_train[indices]
        print(f"Using {sample_size} training samples")
    else:
        X_train_sample = X_train
        y_train_sample = y_train
    
    # Train the model
    clf = DecisionTreeClassifier(
        criterion=criterion_name,
        max_depth=max_depth,
        min_samples_split=20,
        min_samples_leaf=10
    )
    
    print("Training model...")
    start_time = time.time()
    clf.fit(X_train_sample, y_train_sample)
    train_time = time.time() - start_time
    
    print(f"Training time: {train_time:.2f} seconds")
    
    # Evaluate on training set
    print("Evaluating on training set...")
    start_time = time.time()
    train_score = clf.score(X_train_sample, y_train_sample)
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
    print(f"  Train predict time:  {train_pred_time:.2f} seconds")
    print(f"  Test predict time:   {test_pred_time:.2f} seconds")
    
    return {
        'criterion': criterion_name,
        'train_accuracy': train_score,
        'test_accuracy': test_score,
        'train_time': train_time,
        'train_pred_time': train_pred_time,
        'test_pred_time': test_pred_time
    }


def run_benchmark(max_depth=10, sample_size=5000):
    """
    Run benchmark experiments on the Census Income dataset.
    """
    print("="*60)
    print("CENSUS INCOME DATASET BENCHMARK")
    print("="*60)
    
    # Download and load data
    train_df, test_df = download_census_data()
    
    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(train_df, test_df)
    
    # Test all three criteria
    criteria = ['information_gain', 'gain_ratio', 'gini_index']
    results = []
    
    print(f"\nConfiguration:")
    print(f"  Max depth: {max_depth}")
    print(f"  Sample size: {sample_size if sample_size else 'full dataset'}")
    
    for criterion in criteria:
        result = evaluate_criterion(
            criterion, X_train, y_train, X_test, y_test,
            max_depth=max_depth, sample_size=sample_size
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
    
    # Download and load data
    train_df, test_df = download_census_data()
    X_train, X_test, y_train, y_test = preprocess_data(train_df, test_df)
    
    # Experiment 1: Effect of max_depth
    print("\n" + "="*60)
    print("Experiment 1: Effect of max_depth")
    print("="*60)
    
    depths = [5, 10, 15, 20]
    criterion = 'information_gain'
    sample_size = 5000
    
    print(f"\nUsing {criterion} with {sample_size} training samples")
    print(f"{'Max Depth':<12} {'Train Acc':<12} {'Test Acc':<12} {'Train Time':<12}")
    print("-" * 60)
    
    for depth in depths:
        result = evaluate_criterion(
            criterion, X_train, y_train, X_test, y_test,
            max_depth=depth, sample_size=sample_size
        )
        print(f"{depth:<12} "
              f"{result['train_accuracy']:<12.4f} "
              f"{result['test_accuracy']:<12.4f} "
              f"{result['train_time']:<12.2f}")
    
    # Experiment 2: Different sample sizes
    print("\n" + "="*60)
    print("Experiment 2: Effect of training set size")
    print("="*60)
    
    sample_sizes = [1000, 2500, 5000, 10000]
    max_depth = 10
    
    print(f"\nUsing {criterion} with max_depth={max_depth}")
    print(f"{'Sample Size':<12} {'Train Acc':<12} {'Test Acc':<12} {'Train Time':<12}")
    print("-" * 60)
    
    for size in sample_sizes:
        result = evaluate_criterion(
            criterion, X_train, y_train, X_test, y_test,
            max_depth=max_depth, sample_size=size
        )
        print(f"{size:<12} "
              f"{result['train_accuracy']:<12.4f} "
              f"{result['test_accuracy']:<12.4f} "
              f"{result['train_time']:<12.2f}")


if __name__ == '__main__':
    # Run basic benchmark
    results = run_benchmark(max_depth=10, sample_size=5000)
    
    # Run detailed experiments
    run_detailed_experiments()
    
    print("\n" + "="*60)
    print("Benchmark completed!")
    print("="*60)
