import numpy as np
import matplotlib.pyplot as plt

# Function to compute the likelihood ratio test
def likelihood_ratio_test(x, mu1, Sigma1, mu0, Sigma0, tau):
    # Compute likelihood ratios
    ratio = np.exp(-0.5 * (x - mu1).T @ np.linalg.inv(Sigma1) @ (x - mu1) + 0.5 * (x - mu0).T @ np.linalg.inv(Sigma0) @ (x - mu0))
    ratio *= np.sqrt(np.linalg.det(Sigma0) / np.linalg.det(Sigma1))
    
    # Apply threshold
    if ratio >= tau:
        return 1  # Class 1
    else:
        return 0  # Class 0

# Function to compute TP and FP for a given threshold
def compute_TP_FP(predictions, ground_truth):
    TP = np.sum(np.logical_and(predictions == 1, ground_truth == 1))
    FP = np.sum(np.logical_and(predictions == 1, ground_truth == 0))
    return TP, FP

# Generate synthetic data (replace this with your data)
# Assume ground truth labels are available
np.random.seed(0)
num_samples = 1000
dimension = 2
mean1 = np.array([0, 0])
cov1 = np.array([[1, 0], [0, 1]])
mean0 = np.array([2, 2])
cov0 = np.array([[1, 0], [0, 1]])
X_class1 = np.random.multivariate_normal(mean1, cov1, num_samples)
X_class0 = np.random.multivariate_normal(mean0, cov0, num_samples)
X = np.vstack((X_class1, X_class0))
print(np.shape(X))
Y_true = np.hstack((np.ones(num_samples), np.zeros(num_samples)))
print(np.shape(Y_true))

# Define the range of thresholds
thresholds = np.linspace(0, 10, 100)

# Initialize arrays to store pD and pF values
pD_values = np.zeros_like(thresholds)
pF_values = np.zeros_like(thresholds)

# Loop over thresholds
for i, tau in enumerate(thresholds):
    # Apply likelihood ratio test for each sample
    Y_pred = np.array([likelihood_ratio_test(x, mean1, cov1, mean0, cov0, tau) for x in X])
    
    # Compute TP and FP
    TP, FP = compute_TP_FP(Y_pred, Y_true)
    
    # Compute pD and pF
    pD_values[i] = TP / np.sum(Y_true == 1)
    pF_values[i] = FP / np.sum(Y_true == 0)

# Plot the ROC curve
plt.plot(pF_values, pD_values)
plt.xlabel('False Positive Rate (pF)')
plt.ylabel('True Positive Rate (pD)')
plt.title('ROC Curve')
plt.grid(True)
plt.show()
