import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import csv

# ===================== HW2 Exercise 1 =====================
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
#print(np.shape(female_data))

# Print the first 10 elements of each column
print("First 10 entries of female BMI:", female_data[:10, 0])
print("First 10 entries of female stature:", female_data[:10, 1])
print("First 10 entries of male BMI:", male_data[:10, 0])
print("First 10 entries of male stature:", male_data[:10, 1])
csv_file.close()

# ===================== HW2 Exercise 2 Part b =====================

# Combine all data, or X-matrix, [3224 x 3]
X = np.vstack((male_data, female_data))
# Add column of 1's as a bias term for the model [bias, BMI, Stature] (intercept)
X = np.hstack((np.ones((X.shape[0], 1)), X))
#print(np.shape(X))

# Create array to classify gender, with male (+1) and female (-1), [3224 x 1]
y = np.concatenate((np.ones((male_data.shape[0],1)), -np.ones((female_data.shape[0],1))))
# print(np.shape(y))

# Solve using normal equations
theta_normal_eq = np.linalg.inv(X.T @ X) @ X.T @ y
print("Optimal Theta: ", theta_normal_eq.flatten())

# Note: The goal of finding the optimal theta is to determine the parameters of a linear model that best separates the two genders into 1 and -1

# ===================== HW2 Exercise 2 Part c =====================

# Define theta to be [2 x 1]
theta = cp.Variable((X.shape[1], 1))

# Define cost and objective function (no constraints placed)
cost = cp.sum_squares(X @ theta - y)
objective = cp.Minimize(cost)

# Define and solve the objective function
problem = cp.Problem(objective)
problem.solve(solver=cp.SCS) #other solver types seem to be depracated or "inaccurate"

# Get the optimal theta
theta_cvxpy = theta.value
print("Optimal Theta (CVXPY):", theta_cvxpy.flatten())

# ===================== HW2 Exercise 2 Part e, f =====================

# Gradient Descent with Exact Line Search (Analytical Solution)
def gradient_descent_exact_line_search(X, y, max_iterations=50000):
    loss_list = []  # list to store training loss at each iteration for graphing purposes
    # Initialize theta
    theta = np.zeros((X.shape[1], 1))
    #tolerance = 1e-6  # For debugging

    for iteration in range(max_iterations + 1):
        # gradient of epsilon_train (calculated by hand in part d)       
        gradient = -2 * X.T @ (y - X @ theta)

        # optimal step size formula 
        # eta = g.T @ g / (g.T @ H @ g)
        # Hessian matrix H = 2 * X.T @ X
        alpha_optimal = (gradient.T @ gradient) / (gradient.T @ (2 * X.T @ X) @ gradient)

        # update theta 
        theta_next = theta - alpha_optimal * gradient

        # compute training loss and store in history
        # loss = (1/len(y)) * np.sum((X @ theta - y) ** 2)
        loss = np.linalg.norm(y - X @ theta, 2)
        loss_list.append(loss)

        # # Check for convergence, for debugging
        # if iteration > 0 and np.abs(loss_list[iteration - 1] - loss_list[iteration]) < tolerance:
        #     print(f"Converged at iteration {iteration} with loss {loss}")
        #     break

        # Print the current iteration and the updated theta at every 10 thousandth iteration, for debugging
        # if iteration % 10000 == 0:
        #     print(f"Iteration {iteration}: Theta = {theta.flatten()}, Optimal Alpha = {alpha_optimal}")
        # if iteration == max_iterations:
        #    print(f"Iteration with Exact Line Search {iteration}: Theta = {theta_next.flatten()}, Optimal Alpha = {alpha_optimal}")

        theta = theta_next

    return theta, loss_list, alpha_optimal



theta_final, loss_list, alpha_optimal = gradient_descent_exact_line_search(X, y)
print(f"Optimal Theta (Exact Line Search) = {theta_final.flatten()}")
#print(alpha_optimal)

# Plot training loss as a function of iteration number
plt.figure(1)
plt.semilogx(range(0, len(loss_list)), loss_list, linewidth=8)
plt.xlabel('Iteration Number')
plt.ylabel('Training Loss')
plt.title('Training Loss vs. Iteration Number')
plt.savefig('2f_training_loss.png')

# # ===================== HW2 Exercise 2 Part g,h =====================

def gradient_descent_momentum(X, y, beta=0.9, max_iterations=50000):
    loss_list = []  # list to store training loss at each iteration for graphing purposes
    # Initialize theta and momentum term
    theta = np.zeros((X.shape[1], 1))
    momentum = np.zeros_like(theta) # gradient at k-1 term
    #tolerance = 0.000000001  # tolerance to break the loop if the optimal theta is found before max_iterations

    for iteration in range(max_iterations + 1):
        
        gradient = -2 * X.T @ (y - X @ theta) #gradient at k-1

        # Update momentum
        momentum = beta * momentum + (1 - beta) * gradient

        # Update theta with momentum, alpha was calculated from exact line search
        theta_next = theta - alpha_optimal * momentum

        # compute training loss and store in history
       # loss = (1/len(y)) * np.sum((X @ theta - y) ** 2)
        loss = np.linalg.norm(y - X @ theta, 2)
        loss_list.append(loss)

        # # Check for convergence
        # if iteration > 0 and np.abs(loss_list[iteration - 1] - loss_list[iteration]) < tolerance:
        #     print(f"Momentum method converged at iteration {iteration} with loss {loss}")
        #     break

        # ##Print the current iteration and the updated theta at every nth iteration, for debugging
        # if iteration % 100 == 0:
        #     print(f"Iteration {iteration}: Theta = {theta.flatten()}, Momentum = {momentum.flatten()}")

        theta = theta_next

    return theta, loss_list

theta_final_momentum, loss_list_momentum = gradient_descent_momentum(X, y)
print(f"Optimal Theta (Momentum Method) = {theta_final_momentum.flatten()}")

# Plot training loss as a function of iteration number
plt.figure(2)
plt.semilogx(range(0, len(loss_list_momentum)), loss_list_momentum, linewidth=8)
plt.xlabel('Iteration Number')
plt.ylabel('Training Loss')
plt.title('Training Loss vs. Iteration Number (Momentum Method)')
plt.savefig('2g_training_loss_momentum.png')

plt.show()