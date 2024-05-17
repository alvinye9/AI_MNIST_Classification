import numpy as np
import matplotlib.pyplot as plt
from scipy.special import eval_legendre
from scipy.optimize import linprog

# Exercise 3 part a

# Given parameters
beta = [-0.001, 0.01, 0.55, 1.5, 1.2]
#print(beta)
epsilon_std_dev = 0.2
epsilon_mean = 0
error_term = np.random.normal(epsilon_mean, epsilon_std_dev, size=50)

N=50
# Generate x values
x = np.linspace(-1, 1, N)

legendre_polynomials = []
# Find Legendre polynomials
#Result: 50 x 5 matrix, cols are the nth Legendre Polynomial evaluated at each value in the array of 50 x values from -1 to 1 
for i in range(len(beta)):
    legendre_polynomials.append(eval_legendre(i,x))
legendre_polynomials = np.transpose(legendre_polynomials) 

# y = L0*b0 + L1*b1 + L2*b2 + L3*b3 + L4*b4
# legendre_polynomials (50x5) * beta (5x1)  + error_term (50x1) should produce (50x1) vector representing y values evaluated WRT x value 
y = np.dot(legendre_polynomials, beta) + error_term


# Scatter plot
plt.figure(1, figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(x, y, label='Generated Data')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()



# Exercise 3 part c

# form of optimal solution to linear regression problem: 
#          beta = (X^T * X)^-1 * X^T * y
X = legendre_polynomials
beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y #should be (5x1) col vector of coefficients
print("Optimal Beta Values: ", beta_hat)
g_beta = np.dot(X,beta_hat)

#===================== Add Outliers =====================
idx = [5, 6] # these are the locations of the outliers
y[idx] = 3 # set the outliers to have a value 5

#Calculations with Outliers 
beta_hat_outliers = np.linalg.inv(X.T @ X) @ X.T @ y 
g_beta_outliers = np.dot(X,beta_hat_outliers)
#=========================================================

# plot y again, on the right subplot, but with outliers
plt.subplot(1, 2, 2)
plt.scatter(x, y, label='Generated Data')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()


# overlay scatter plot with prediceted curve
plt.subplot(1, 2, 1)
plt.plot(x, g_beta, label='Predicted Curve (no outliers)', color = 'red')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression Problem and Solution')
plt.legend()

#overlay right scatter plot
plt.subplot(1, 2, 2)
plt.plot(x, g_beta_outliers, label='Predicted Curve (with outliers)', color = 'red')
plt.xlabel('x')
plt.ylabel('y')
plt.title('With Outliers at y=3')
plt.legend()
plt.ylim(-1.5, 4 )
plt.savefig('quiz1_3_Linear_Regression')

# # Exercise 3 part e
# A = np.block([[X, -np.identity(np.shape(X)[0])], [-X, -np.identity(np.shape(X)[0])]]) #(100x55)
# b = np.block([y,-1*y]).T #(100x1)
# c = np.block([np.zeros(np.shape(X)[0]), np.ones(np.shape(X)[1])]).T #(55x1)

X = np.column_stack((np.ones(N), x, x**2, x**3, x**4))
A = np.vstack((np.hstack((X, -np.eye(N))), np.hstack((-X, -np.eye(N)))))
b = np.hstack((y,-y))
c = np.hstack((np.zeros(5), np.ones(N)))


# Solve the linear programming problem 
# Goal: Minimize x for c^T * x subject to Ax<=b
result = linprog(c, A_ub = A, b_ub = b, bounds=(None, None), method="revised simplex")
# beta_hat_lp = result.x[:5] # x is a matrix of coefficient values followed by u-values, keep first d points
beta_hat_lp = result.x[:np.shape(X)[1]]  # Keep the first d=5 elements which are coefficient (beta) values, followed by u values

print(beta_hat)
print(beta_hat_outliers)
print(beta_hat_lp)

g_beta_lp = np.dot(X,beta_hat_lp)

# Overlay the predicted curve with the scatter plot
plt.figure(2)
plt.scatter(x, g_beta_lp, alpha=0.5, label='Linear Programming Results (Outliers at y=3)')
plt.plot(x, g_beta, label='Predicted Curve (no outliers)', color = 'red')
plt.xlabel('x')
plt.ylabel('y')
plt.title("Linear Programming Solution vs Corrupted Data")
plt.legend()
plt.ylim(-1.5, 4)
plt.savefig('quiz1_3_linear_programming')


plt.show()
