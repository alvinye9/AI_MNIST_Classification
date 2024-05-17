import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from scipy.stats import multivariate_normal

#========================================== HW4 Exercise 2 part b ==========================================
class1_data = np.loadtxt('quiz4_class1.txt')
class0_data = np.loadtxt('quiz4_class0.txt')

X = np.vstack((class1_data, class0_data))
y = np.vstack((np.ones((class1_data.shape[0], 1)), np.zeros((class0_data.shape[0], 1))))

X_with_bias = np.hstack((X, np.ones((X.shape[0], 1))))  # Add bias term
# print(np.shape(X_with_bias)) #[100 x 3]
theta = cp.Variable((X_with_bias.shape[1], 1))

lambda_ = 0.0001
N = X.shape[0]
d = X.shape[1]

# data fidelity and regularization terms
data_fidelity = - (1 / N) * ( cp.sum(cp.multiply(y, X_with_bias @ theta)) - cp.sum(cp.log_sum_exp( cp.hstack([np.zeros((N,1)), X_with_bias @ theta]), axis=1)) ) #axis=1 allows log_sum_exp to sum row-by-row
regularization = lambda_ * cp.norm(theta, 2)**2  # Exclude bias term from regularization

loss = data_fidelity + regularization
prob = cp.Problem(cp.Minimize(loss))
prob.solve(solver=cp.SCS)

#optimal theta values
w_2, w_1, w_0 = theta.value
print("Optimal theta:", theta.value)

#========================================== HW4 Exercise 2 part c ==========================================
plt.scatter(class1_data[:, 0], class1_data[:, 1], color='blue', label='Class 1')
plt.scatter(class0_data[:, 0], class0_data[:, 1], color='red', label='Class 0')

# Plot the decision boundary
x_values = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)
y_values = -(w_1 * x_values + w_0) / w_2
plt.figure(1)
plt.plot(x_values, y_values, color='green', label='Decision Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.title('Logistic Regression with Regularization')
plt.savefig("2c_logistic_regression.png")

#========================================== HW4 Exercise 2 part d ==========================================
# Implement Bayesian Decision Rule P(Y|X) = (P(X|Y) * P(Y)) / P(X)
# P(Y|X): posterior
# P(X|Y): likelihood
# P(Y): prior
# P(X): marginal

x1_range = np.linspace(-5, 10, 100)
x2_range = np.linspace(-5, 10, 100)
x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
testing_sites = np.column_stack([x1_grid.ravel(), x2_grid.ravel()])

mu1 = np.mean(class1_data, axis=0) #[2 x 1]
mu0 = np.mean(class0_data, axis=0)
cov1 = np.cov(class1_data.T)
cov0 = np.cov(class0_data.T)
# print(np.shape(cov1)) # [2 x 2]

# # Calculate likelihood, assuming multivariate normal distribution
pdf1 = multivariate_normal.pdf(testing_sites, mean=mu1, cov=cov1)
pdf0 = multivariate_normal.pdf(testing_sites, mean=mu0, cov=cov0)

# Prior calculations (should be 0.5, 0.5)
prior1 = class1_data.shape[0] / X_with_bias.shape[0]
prior0 = class0_data.shape[0] / X_with_bias.shape[0]

# Calculate posterior probabilities, can omit denominator P(X)
posterior1 = pdf1 * prior1
posterior0 = pdf0 * prior0
# print(np.shape(posterior1)) #[10000 x 1]

# Apply Bayesian decision rule
predictions = []
# Iterate over each row index
for i in range(len(posterior1)):
    if posterior1[i] < posterior0[i]: # < seems to fix the classification, why?
        predictions.append(1)
    else:
        predictions.append(0)

predictions = np.array(predictions)
predictions = predictions.reshape(x1_grid.shape)
# print(np.shape(predictions))

# Plot decision boundary using contours
plt.figure(2)
plt.contourf(x1_grid, x2_grid, predictions, cmap=plt.cm.coolwarm, alpha=0.3)
plt.scatter(class1_data[:, 0], class1_data[:, 1], color='blue', label='Class 1')
plt.scatter(class0_data[:, 0], class0_data[:, 1], color='red', label='Class 0')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Boundary using Bayesian Decision Rule')
plt.colorbar(label='Predicted Class')
plt.legend()
plt.savefig("2d_bayesian.png")
plt.show()


