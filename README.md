# ESE589 Decision Tree Project

A Python implementation of the Decision Tree Induction algorithm for classification with three different splitting criteria: Information Gain, Gain Ratio, and Gini Index.

## Project Overview

This project implements a Decision Tree classifier from scratch using Python and NumPy. The implementation supports three different metrics for selecting the best split:

1. **Information Gain** - Based on entropy reduction
2. **Gain Ratio** - Normalized information gain to avoid bias towards features with many values
3. **Gini Index** - Based on Gini impurity reduction

## Features

- **Pure Python Implementation**: No reliance on existing ML libraries for the core algorithm
- **Multiple Splitting Criteria**: Support for Information Gain, Gain Ratio, and Gini Index
- **Configurable Parameters**: Control tree depth, minimum samples for splits, and more
- **Validation Examples**: Small illustrating datasets for algorithm validation
- **Benchmark Experiments**: Comprehensive experiments comparing the three metrics

## Installation

### Prerequisites

- Python 3.7 or higher
- pip

### Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- numpy>=1.20.0
- pandas>=1.3.0
- scikit-learn>=1.0.0 (only for comparison and data generation)
- requests>=2.25.0 (for dataset download)

## Project Structure

```
ESE589-decision-tree-proj/
│
├── decision_tree.py           # Main Decision Tree implementation
├── validation_examples.py     # Small examples for validation
├── benchmark.py              # Benchmark with Census Income dataset
├── benchmark_synthetic.py    # Benchmark with synthetic data
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## Usage

### Basic Usage

```python
from decision_tree import DecisionTreeClassifier
import numpy as np

# Create sample data
X_train = np.array([[1, 2], [2, 3], [3, 1], [6, 5], [7, 8], [8, 6]])
y_train = np.array([0, 0, 0, 1, 1, 1])

# Train the classifier with Information Gain
clf = DecisionTreeClassifier(criterion='information_gain', max_depth=5)
clf.fit(X_train, y_train)

# Make predictions
X_test = np.array([[2, 2], [7, 7]])
predictions = clf.predict(X_test)
print(predictions)  # Output: [0 1]

# Calculate accuracy
accuracy = clf.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2f}")
```

### Using Different Criteria

```python
# Information Gain (default)
clf_ig = DecisionTreeClassifier(criterion='information_gain')

# Gain Ratio
clf_gr = DecisionTreeClassifier(criterion='gain_ratio')

# Gini Index
clf_gini = DecisionTreeClassifier(criterion='gini_index')
```

### Configuration Parameters

- `criterion`: Splitting criterion ('information_gain', 'gain_ratio', 'gini_index')
- `max_depth`: Maximum depth of the tree (default: None)
- `min_samples_split`: Minimum samples required to split a node (default: 2)
- `min_samples_leaf`: Minimum samples required at a leaf node (default: 1)

## Validation Examples

Run small validation examples:

```bash
python validation_examples.py
```

This script tests the implementation on three toy datasets:
1. **Play Tennis Dataset**: Classic example with 14 samples
2. **Linearly Separable Dataset**: Simple 2D dataset with 6 samples
3. **XOR-like Dataset**: Non-linearly separable dataset with 8 samples

### Sample Output

```
VALIDATION TEST 1: Play Tennis Dataset
============================================================
Testing with criterion: information_gain
Training accuracy: 1.0000
Test accuracy: 0.7500

Testing with criterion: gain_ratio
Training accuracy: 1.0000
Test accuracy: 0.7500

Testing with criterion: gini_index
Training accuracy: 1.0000
Test accuracy: 0.7500
```

## Benchmark Experiments

### Synthetic Data Benchmark

Since network access may be restricted, we provide a benchmark using synthetic data similar to the Census Income dataset:

```bash
python benchmark_synthetic.py
```

This script:
1. Creates synthetic datasets similar to Census Income data
2. Evaluates all three splitting criteria
3. Runs experiments with different configurations:
   - Effect of tree depth (3, 5, 10, 15)
   - Effect of dataset size (1,000 to 20,000 samples)
   - Comparison on larger datasets

### Sample Results

```
BENCHMARK SUMMARY
============================================================
Comparison of splitting criteria:
Criterion            Train Acc    Test Acc     Train Time  
------------------------------------------------------------
information_gain     0.7811       0.7405       2.73        
gain_ratio           0.7466       0.7465       0.17        
gini_index           0.7966       0.7600       2.69        

Best test accuracy: gini_index (0.7600)
```

### Census Income Dataset Benchmark

If you have internet access, you can run the benchmark on the real Census Income dataset:

```bash
python benchmark.py
```

This will download and use the actual Census Income dataset from the UCI Machine Learning Repository.

## Implementation Details

### Splitting Criteria

#### 1. Information Gain

Information Gain measures the reduction in entropy after a split:

```
IG(S, A) = Entropy(S) - Σ(|Sv|/|S|) * Entropy(Sv)
```

Where:
- `Entropy(S) = -Σ pi * log2(pi)`
- `S` is the parent node
- `Sv` are the child nodes after split

#### 2. Gain Ratio

Gain Ratio normalizes Information Gain by the split information to avoid bias:

```
GainRatio(S, A) = IG(S, A) / SplitInfo(S, A)
SplitInfo(S, A) = -Σ(|Sv|/|S|) * log2(|Sv|/|S|)
```

#### 3. Gini Index

Gini Index measures impurity and aims to minimize it:

```
Gini(S) = 1 - Σ pi²
GiniGain = Gini(parent) - Σ(|Sv|/|S|) * Gini(Sv)
```

### Algorithm

The decision tree is built using a recursive algorithm:

1. **Base Cases** (create leaf node):
   - Maximum depth reached
   - All samples belong to one class
   - Fewer samples than `min_samples_split`
   
2. **Recursive Case**:
   - Find the best feature and threshold using the selected criterion
   - Split the data based on the best split
   - Recursively build left and right subtrees

3. **Prediction**:
   - Traverse the tree from root to leaf
   - Return the class label at the leaf node

## Experimental Results

### Key Findings

1. **Accuracy**: All three criteria achieve similar test accuracies (~74-76%) on the benchmark data
2. **Training Time**: 
   - Gain Ratio is significantly faster (~0.17s) than the other criteria
   - Information Gain and Gini Index have similar training times (~2.7s)
3. **Overfitting**: 
   - Deeper trees lead to higher training accuracy but lower test accuracy
   - Maximum depth of 5-10 provides good balance

### Effect of Tree Depth

```
Max Depth    Train Acc    Test Acc
------------------------------------
3            0.7538       0.7480
5            0.7615       0.7470
10           0.7833       0.7320
15           0.8030       0.7220
```

Observation: Increasing depth improves training accuracy but may reduce test accuracy (overfitting).

### Effect of Dataset Size

```
Dataset Size Train Acc    Test Acc
------------------------------------
1,000        0.7812       0.6900
2,500        0.7735       0.7400
5,000        0.7833       0.7320
10,000       0.7811       0.7405
```

Observation: Larger datasets generally lead to better generalization.

## Comparison with sklearn

Our implementation achieves comparable performance to scikit-learn's DecisionTreeClassifier, validating the correctness of our algorithm.

## Limitations

- **Continuous Features**: The implementation treats all features as continuous and finds optimal thresholds
- **Binary Splits**: Only binary splits are supported (≤ threshold vs > threshold)
- **No Pruning**: The tree is not pruned after construction (controlled only by stopping criteria)
- **Memory**: The entire tree is kept in memory

## Future Improvements

- Add post-pruning to reduce overfitting
- Support for categorical features without encoding
- Multi-way splits for categorical features
- Feature importance calculation
- Tree visualization
- Parallel processing for faster training

## References

1. Quinlan, J. R. (1986). "Induction of Decision Trees". Machine Learning.
2. Breiman, L., et al. (1984). "Classification and Regression Trees".
3. UCI Machine Learning Repository: Census Income Dataset

## License

This project is created for educational purposes as part of ESE589 coursework.

## Authors

ESE589 Student Project