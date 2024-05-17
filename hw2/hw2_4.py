import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import csv

# Reading csv file for male data
with open("male_train_data.csv", "r") as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    #skip first row (heading)
    next(reader) 
    # Processing the data into usable form
    male_data = []
    for row in reader:
        # Normalize male_stature_mm and male_bmi
        stature_mm = float(row[2]) / 1000
        bmi = float(row[1]) / 10
        male_data.append([bmi, stature_mm])

csv_file.close()
# Reading csv file for female data
with open("female_train_data.csv", "r") as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    next(reader)
    female_data = []
    for row in reader:
        # Normalize female_stature_mm and female_bmi
        stature_mm = float(row[2]) / 1000
        bmi = float(row[1]) / 10
        female_data.append([bmi, stature_mm])

# Convert the data into NumPy arrays
male_data = np.array(male_data)
female_data = np.array(female_data)
csv_file.close()

# Combine all data, or X-matrix, add a col of 1's [3224 x 3]
X = np.vstack((male_data, female_data))
X = np.hstack((np.ones((X.shape[0], 1)), X))
#print(np.shape(X))

# Create array to classify gender, with male (+1) and female (-1), [3224 x 1]
y = np.concatenate((np.ones((male_data.shape[0],1)), -np.ones((female_data.shape[0],1))))
# print(np.shape(y))

# Solve using numpy functions, [3 x 1]
theta = np.linalg.inv(X.T @ X) @ X.T @ y
#print("Optimal Theta: ", theta.flatten())

# ===================== HW2 Exercise 4 =====================

# helper functions to solve the three optimization functions
def optimal_theta_lambda(X, y, lambd):
    theta = cp.Variable((X.shape[1], 1))
    objective = cp.Minimize(cp.norm(X @ theta - y, 2)**2 + lambd * cp.norm(theta, 2)**2)
    problem = cp.Problem(objective)
    problem.solve(solver=cp.SCS)
    return theta.value

def optimal_theta_alpha(X, y, alpha):
    theta = cp.Variable((X.shape[1], 1))
    objective = cp.Minimize(cp.norm(X @ theta - y, 2)**2)
    constraints = [cp.norm(theta, 2) <= alpha]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCS)
    return theta.value

def optimal_theta_epsilon(X, y, epsilon):
    theta = cp.Variable((X.shape[1], 1))
    objective = cp.Minimize(cp.norm(theta, 2)**2)
    constraints = [cp.norm(X @ theta - y, 2) <= epsilon]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCS)
    return theta.value

# generate values to graph
lambd_values = np.arange(0.1, 10, 0.1)
norm_X_theta_minus_y_sq = []
norm_X_theta_minus_y_sq_vs_norm_theta_sq = []
norm_theta_sq_values = []

# Solve the optimization problems for each lambda
for lambd in lambd_values:
    # Solve θλ
    theta_lambda = optimal_theta_lambda(X, y, lambd)

    # Calculate norm(Xtheta - y)^2
    norm_X_theta_minus_y_sq.append(np.linalg.norm(X @ theta_lambda - y)**2)

    # Calculate norm(theta)^2
    norm_theta_sq_values.append(np.linalg.norm(theta_lambda)**2)



plt.figure(1, figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(norm_theta_sq_values, norm_X_theta_minus_y_sq, label='||Xtheta - y||^2 vs ||theta||^2')
plt.xlabel('||theta||^2')
plt.ylabel('||Xtheta - y||^2')
plt.title('||Xtheta - y||^2 vs ||theta||^2')

plt.subplot(1, 3, 2)
plt.plot(lambd_values, norm_X_theta_minus_y_sq, label='||Xtheta - y||^2 vs lambda')
plt.xlabel('lambda')
plt.ylabel('||Xtheta - y||^2')
plt.title('||Xtheta - y||^2 vs lambda')

plt.subplot(1, 3, 3)
plt.plot(lambd_values, norm_theta_sq_values, label='||theta||^2 vs lambda')
plt.xlabel('lambda')
plt.ylabel('||theta||^2')
plt.title('||theta||^2 vs lambda')

plt.tight_layout()
plt.savefig('4a_three_optimization_problems')



plt.show()
