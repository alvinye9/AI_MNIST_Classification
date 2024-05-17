import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

# Load data from text files
class1_data = np.loadtxt('homework4_class1.txt')
class0_data = np.loadtxt('homework4_class0.txt')

# Combine the data and labels
X = np.vstack((class1_data, class0_data))
y = np.vstack((np.ones((class1_data.shape[0], 1)), np.zeros((class0_data.shape[0], 1)))) 

# Normalize features, is this what they mean?
# X_normalized = X / np.linalg.norm(X, axis=0)

# homework sheet doesnt seem to change anything, so ill keep X as is for now
X_normalized = X 

# Add ones column with the original feature matrix to account for intercept term in theta
ones_column = np.ones((X.shape[0], 1))
X_with_intercept = np.hstack((ones_column, X))

# Define parameters
N, d = X_with_intercept.shape
lambda_ = 0.0001

# Define optimization variable
theta = cp.Variable((d, 1))

#Variable Shapes:
# X: [100 x 3]
# y: [100 x 1]
# theta: [3 x 1]

# print( cp.vstack([np.zeros((N, 1)), -cp.multiply(y, X_with_intercept @ theta)]).shape) #[200 x 1], for debugging

# Define first part of data fidelity term 
first_term = cp.sum(cp.multiply(y, X_with_intercept)  @ theta)

# Define logistic term, use log_sum_exp() to represent log(e^0 + e^(theta^T * x_n))
# log_term = cp.sum(cp.log_sum_exp(cp.vstack([np.zeros((N, 1)), +cp.multiply(y, X_with_intercept @ theta)])))
log_term = cp.sum(cp.log_sum_exp(cp.vstack([np.zeros((N, 1)), X_with_intercept @ theta])))



data_fidelity = - (1 / N) * (first_term - log_term)

# Define regularization term
regularization = lambda_ * cp.norm(theta, 2)**2

# Define objective function
objective = data_fidelity + regularization

# Define optimization problem
problem = cp.Problem(cp.Minimize(objective))

# Solve the problem
problem.solve(solver=cp.SCS)

# Extract optimal theta values
theta_optimal = theta.value

print("Optimal theta:", theta_optimal) # [theta_2, theta_1, theta_0]^T


#========================================== HW4 Exercise 2 part c ==========================================

# Extract coefficients from the optimal theta values (idk why the flipped theta values seem to do better)
theta_0 = theta_optimal[0]
theta_1 = theta_optimal[1]
theta_2 = theta_optimal[2]

# theta_0 = theta_optimal[2]
# theta_1 = theta_optimal[1]
# theta_2 = theta_optimal[0]

# Plot two classes
plt.scatter(class1_data[:, 0], class1_data[:, 1], color='blue', label='Class 1')
plt.scatter(class0_data[:, 0], class0_data[:, 1], color='red', label='Class 0')

# Plot the decision boundary
x1_values = np.linspace(min(X[:, 0]), max(X[:, 0]), 100)
x2_values = -(theta_0 + theta_1 * x1_values) / theta_2
plt.plot(x1_values, x2_values, color='green', label='Decision Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.savefig("2c_logistic_regression.png")

plt.show()
