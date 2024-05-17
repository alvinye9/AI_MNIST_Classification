
import numpy as np

# Simulated values of V1 and known true mean mu1
V1_values = np.array([0.48, 0.52, 0.55, 0.47, 0.49])
mu1 = 0.5

# Specific value of epsilon
epsilon = 0.02

count_v1 = np.sum(np.abs(V1_values - mu1) > epsilon)
print(np.abs(V1_values - mu1)> epsilon) #array of true or false
print("Count of instances where |V1 - mu1| > epsilon:", count_v1)
