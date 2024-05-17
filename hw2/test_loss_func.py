import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import csv

A = np.array([1,2,3,4]).T
B = np.array([1,2,0,324]).T

print(A)
print(B)
print(A-B)
print((A-B)**2)

print((1/len(A)) * np.sum((A - B)**2))



# # Gradient Descent with Exact Line Search (using CVXPY)
# def gradient_descent_exact_line_search(X, y, max_iterations=50000):
#     loss_list = []  # list to store training loss at each iteration for graphing purposes
#     # Initialize theta
#     theta = np.zeros((X.shape[1], 1))
#     tolerance = 0.0000000000000001  # tolerance to break loop if optimal theta is found before 50000 iterations

#     for iteration in range(max_iterations + 1):
#         # gradient of epsilon_train (calculated by hand in part d)
#         gradient = -2 * X.T @ (y - X @ theta)

#         # optimal step size using exact line search
#         alpha_optimal = cp.Variable()
#         objective_alpha = cp.Minimize(cp.sum_squares(X @ (theta - alpha_optimal * gradient) - y))
#         problem_alpha = cp.Problem(objective_alpha)
#         problem_alpha.solve()
#         alpha_optimal_value = alpha_optimal.value.item()

#         # update theta 
#         theta_next = theta - alpha_optimal_value * gradient

#         # compute training loss and store in history, loss is calculated as difference between the y calculated from optimal theta, and actual y (-1 and 1 gender classification)
#         loss = (1/len(y)) * np.sum((X @ theta - y) ** 2)
#         loss_list.append(loss)

#         # Check to see if loss is becoming constant, if so it means answer converged
#         # Original idea was to simply see if loss was below tolerance, but for some reason loss is hovering around 0.92
#         if iteration > 0 and np.abs(loss_list[iteration - 1] - loss_list[iteration]) < tolerance:
#             # print(f"Exact Line Search converged at iteration {iteration} with loss {loss}") #for debugging
#             break

#         # ##Print the current iteration and the updated theta at every nth iteration, for debugging
#         # if iteration % 100 == 0:
#         #     print(f"Iteration {iteration}: Theta = {theta.flatten()}, Optimal Alpha = {alpha_optimal_value}, Loss = {loss}")

#         theta = theta_next

#     return theta, loss_list, alpha_optimal_value
