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

# Calcuate Kernel Matrix
# K(x,x_n) = exp(-||x-x_n||^2 /h)
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
loss = cp.psd_wrap(data_fidelity + regularization)

prob = cp.Problem(cp.Minimize(loss))
prob.solve(solver=cp.SCS)
print("First two elements of the optimal alpha:", alpha.value[:2])

#========================================== HW4 Exercise 3 part d ==========================================
# plot decision boundary like 2d
x1_range = np.linspace(-5, 10, 100)
x2_range = np.linspace(-5, 10, 100)
x1_grid, x2_grid = np.meshgrid(x1_range, x2_range) #[100 x 100] meshgrid with x1 and x2 values per grid, each (x1,x2) is a new data point
# print(x1_grid.shape) #[100 x 100]
testing_sites = np.column_stack([x1_grid.ravel(), x2_grid.ravel()]) #[10000 x 2]

# g_theta(x) = ∑ αK(x,x_n)
# K(x,x_n) = exp(-||x-x_n||^2 /h)
# Use previous formula for calculating Kernel Matrix, simpler than iterating through K
pdf1 = np.exp(-np.linalg.norm(testing_sites[:, None] - X, axis=2)**2 / h).dot(alpha.value)  #axis=2 allows norm() to calculate the magnitude of each vector difference
pdf0 = np.exp(-np.linalg.norm(testing_sites[:, None] - X, axis=2)**2 / h).dot(1 - alpha.value) 
# print(pdf1.shape) #[10000 x 1]

predictions = np.where(pdf1 < pdf0, 1, 0) #[10000 x 1]
# print(predictions.shape)
predictions = predictions.reshape(x1_grid.shape)
# print(predictions.shape)

plt.figure(3)
plt.contourf(x1_grid, x2_grid, predictions, cmap=plt.cm.coolwarm, alpha=0.3)
plt.scatter(class1_data[:, 0], class1_data[:, 1], color='blue', label='Class 1')
plt.scatter(class0_data[:, 0], class0_data[:, 1], color='red', label='Class 0')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Boundary using Kernel Logistic Regression')
plt.colorbar(label='Predicted Class')
plt.legend()
plt.savefig("quiz4_kernel_logistic_regression.png")

plt.show()


