import numpy as np
import matplotlib.pyplot as plt

# Exercise 2 Part a
# Define the simplified PDF function
def f_X(x1, x2):
    return  (1 / np.sqrt(12 * np.pi))  * np.exp(-1/6 * (2*x1**2 - 2*x1*x2 + 4*x1 + 2*x2**2   - 20*x2 + 56))

# Create a grid of values for x1 and x2
x1 = np.linspace(-1, 5, 100) # 100 element array
x2 = np.linspace(0, 10, 100) # 100 element array

X1, X2 = np.meshgrid(x1, x2) # two 100x100 element arrays


# Calculate the PDF values for each point in the mesh grid
pdf_values = f_X(X1, X2)

# Contour plot
plt.figure(1)
contour = plt.contour(X1, X2, pdf_values)
plt.colorbar(contour, label='PDF')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Contour Plot of $f_X(x)$')
plt.savefig('2a_contour_plot')

# Exercise 2 Part c

# # covariance and mean (given)
# cov = [[2, 1], [1, 2]] 
# mean = [2, 6]

# covariance and mean for 2D Standard Normal Distribution X~N(0,I)
cov_x = [[1, 0], [0, 1]] 
mean_x = [0, 0] #col vector

# Draw random samples from 2D standard normal distribution
# np.random.multivariate_normal returns 5000 x 2 matrix of x1, x2 points, standard form of X is X = [X_1; X_2], so we must transpose
X = np.random.multivariate_normal(mean_x, cov_x, 5000).T

X1, X2 = X #conveniently define X1, X2 as first and second row, resp.

# Scatter plot
plt.figure(2)
plt.scatter(X1, X2, alpha=0.5) #make points transparent to easily see density
plt.title('Scatter Plot of Random Samples from X~N(0,I)')
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.savefig('2c_2D_standard_normal_distribution')

# covariance and mean for random variable Y
cov_y = [[2, 1], [1, 2]] 
mean_y = [2, 6]

# Eigen-decomposition
eigenvalues, eigenvectors = np.linalg.eig(cov_y)

# D = np.array([[eigenvalues[0], 0], [0, eigenvalues[1]]]) # diag matrix from PDP^-1
# P = eigenvectors # eigenvectors are automatically normalized
# A = np.matmul(P, np.sqrt(D)) # matrix multiplication

D = [[eigenvalues[0], 0], [0, eigenvalues[1]]] # diag matrix from PDP^-1
P = eigenvectors # eigenvectors are automatically normalized
A = P @ np.sqrt(D) # matrix multiplication

A_calculated = [[1/np.sqrt(2), np.sqrt(3/2)], [-1/np.sqrt(2), np.sqrt(3/2)]] # calculated by hand from 2b, for comparison
b = mean_y # calculated by hand from 2b
b = np.reshape(b, (2, 1))

print(A)
print(A_calculated)

# Apply the transformation to the 5000 data points
Y = A @ X  + b #should theoretically be the same normal distribution as N(mean_y, cov_y)
Y1, Y2 = Y

# Theoretical Scatter plot of Y based on calculations
plt.figure(3,figsize=(12, 6))  
plt.subplot(1, 2, 1)  
plt.scatter(Y1, Y2, alpha=0.5) #make points transparent to easily see density
plt.title('Theoretical Scatter Plot of Random Samples from Y~N([2; 6],[2 1; 1 2])')
plt.xlabel('y_1')
plt.ylabel('y_2')

# Actual Scatter Plot of Y for reference
Y = np.random.multivariate_normal(mean_y, cov_y, 5000).T
Y1, Y2 = Y
plt.subplot(1, 2, 2)  
plt.scatter(Y1, Y2, alpha=0.5) #make points transparent to easily see density
plt.title('Actual ')
plt.xlabel('y_1')
plt.ylabel('y_2')
plt.savefig('2c_theoretical_vs_actual_Y')

plt.show()




