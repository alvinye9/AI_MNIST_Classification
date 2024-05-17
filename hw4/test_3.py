import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

#========================================== HW4 Exercise 3 part a ==========================================
class1_data = np.loadtxt('quiz4_class1.txt')
class0_data = np.loadtxt('quiz4_class0.txt')
X = np.vstack((class1_data, class0_data))
y = np.vstack((np.ones((class1_data.shape[0], 1)), np.zeros((class0_data.shape[0], 1)))) #[100 x 1]

h = 1
K = np.zeros((X.shape[0], X.shape[0]))
# print(X.shape[0])
# print(X[1])

# Calculate the kernel matrix using the given formula
for m in range(X.shape[0]):
    for n in range(X.shape[0]):
        norm_squared = np.linalg.norm(X[m] - X[n])**2
        K[m, n] = np.exp(-norm_squared / h)  
print("K[47:52, 47:52]: ")
print(K[47:52, 47:52])
# print(np.shape(K)) #[100 x 100]
# ans = np.exp(-1*np.linalg.norm(X[32]-X[20])**2) #for debugging
# print(ans)
# print(K[32,20])

#========================================== HW4 Exercise 3 part c ==========================================
alpha = cp.Variable((K.shape[1], 1)) #[100 x 1]
lambda_ = 0.01
N = K.shape[0]

data_fidelity =  - (1 / N) * ((y.T @ K @ alpha) - cp.sum(cp.log_sum_exp(cp.hstack([np.zeros((N, 1)), K @ alpha]), axis=1)))
regularization = lambda_ * cp.quad_form(alpha, K)
loss = data_fidelity + regularization

prob = cp.Problem(cp.Minimize(loss))
prob.solve(solver=cp.SCS)
print("First two elements of the optimal alpha:", alpha.value[:2])

#========================================== HW4 Exercise 3 part d ==========================================
plt.scatter(class1_data[:, 0], class1_data[:, 1], color='blue', label='Class 1')
plt.scatter(class0_data[:, 0], class0_data[:, 1], color='red', label='Class 0')

# Plot the decision boundary
# The decision boundary is where the output of the classifier is zero
# Compute the output of the classifier for a grid of points and plot the contour where the output is zero
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# Flatten the meshgrid coordinates and compute the kernel values for each point
grid_points = np.c_[xx.ravel(), yy.ravel()]
K_grid = np.zeros((grid_points.shape[0], X.shape[0]))
for i in range(X.shape[0]):
    norm_squared = np.linalg.norm(grid_points - X[i], axis=1)**2
    K_grid[:, i] = np.exp(-norm_squared / h)

# Compute the output of the classifier for the grid points
output = K_grid @ alpha.value
Z = np.sign(output.reshape(xx.shape))

# Plot the contour where the output is zero
plt.contour(xx, yy, Z, colors='k', levels=[0], linestyles=['-'])
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Scatter plot with Decision Boundary')
plt.legend()

plt.show()




# # #========================================== HW4 Exercise 3 part d ==========================================
# # Define grid of testing sites
# x1_range = np.linspace(-5, 10, 100)
# x2_range = np.linspace(-5, 10, 100)
# x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
# testing_sites = np.column_stack([x1_grid.ravel(), x2_grid.ravel()])

# # Calculate class-conditional densities for each class
# pdf1 = np.exp(-np.linalg.norm(testing_sites[:, None] - X, axis=2)**2 / h).dot(alpha.value)  # Gaussian (RBF) kernel
# pdf0 = np.exp(-np.linalg.norm(testing_sites[:, None] - X, axis=2)**2 / h).dot(1 - alpha.value)  # Gaussian (RBF) kernel

# # Apply the Bayesian decision rule
# predictions = np.where(pdf1 < pdf0, 1, 0)
# predictions = predictions.reshape(x1_grid.shape)

# # Plot decision boundary using contours
# plt.figure(figsize=(8, 6))
# plt.contourf(x1_grid, x2_grid, predictions, cmap=plt.cm.coolwarm, alpha=0.3)
# plt.scatter(class1_data[:, 0], class1_data[:, 1], color='blue', label='Class 1')
# plt.scatter(class0_data[:, 0], class0_data[:, 1], color='red', label='Class 0')
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.title('Decision Boundary using Kernel Logistic Regression')
# plt.colorbar(label='Predicted Class')
# plt.legend()
# plt.show()
