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

# ===================== HW2 Exercise 3 =====================

# Extracting theta parameters
theta_0, theta_1, theta_2 = theta.flatten()

# Plot the Training Data Points
plt.figure(1)
plt.scatter(male_data[:, 0], male_data[:, 1], color='blue', label='Male')  # Male data points
plt.scatter(female_data[:, 0], female_data[:, 1], color='red', label='Female')  # Female data points

# Get min and max x1 (BMI/10) values
min_bmi = np.min(X[:, 1])
max_bmi = np.max(X[:, 1])
# print(min_bmi)
# print(max_bmi)

# Decision Boundary
x1_values = np.linspace(min_bmi, max_bmi, 100)
x2_values = (-theta_0 - theta_1 * x1_values) / theta_2  # Calculate x2 by setting g=0
# print(x1_values)
# print(x2_values)
plt.plot(x1_values, x2_values, color='green', label='Decision Boundary')
plt.xlabel('BMI/10')
plt.ylabel('Stature [m]')
plt.title('Visualization of Classifier')
plt.legend()
plt.grid(True)
plt.savefig('3a_training_data_points.png')

# ===================== HW2 Exercise 3, Part b =====================

# Reading csv file for male data
with open("male_test_data.csv", "r") as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    #skip first row (heading)
    next(reader) 
    # Processing the data into usable form
    male_test_data = []
    for row in reader:
        # Normalize male_stature_mm and male_bmi
        stature_mm = float(row[2]) / 1000
        bmi = float(row[1]) / 10
        male_test_data.append([bmi, stature_mm])

csv_file.close()

# Reading csv file for female data
with open("female_test_data.csv", "r") as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    next(reader)
    female_test_data = []
    for row in reader:
        # Normalize female_stature_mm and female_bmi
        stature_mm = float(row[2]) / 1000
        bmi = float(row[1]) / 10
        female_test_data.append([bmi, stature_mm])

# Convert the data into NumPy arrays
male_test_data = np.array(male_test_data)
female_test_data = np.array(female_test_data)
csv_file.close()
# print(np.shape(male_test_data))

X_test = np.vstack((male_test_data, female_test_data))
X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
y_test = np.concatenate((np.ones((male_data.shape[0],1)), -np.ones((female_data.shape[0],1))))

male_test_data = np.hstack((np.ones((male_test_data.shape[0], 1)), male_test_data))
female_test_data = np.hstack((np.ones((female_test_data.shape[0], 1)), female_test_data))

# Define the classification function based on the linear model
def predict_label(x, theta):
    return np.sign(np.dot(x, theta))


# Predict labels for male and female test data
male_predictions = predict_label(male_test_data, theta)
female_predictions = predict_label(female_test_data, theta)

# Calculate Type 1 error (False Alarm/ False Positive), Percentage of female samples misclassified as male
type1_error = np.mean(female_predictions != -1) * 100  
print("Type 1 error (False Alarm) % female misclassified as male:", type1_error, "%")

# Calculate Type 2 error (Miss/ False Negative), Percentage of male samples misclassified as female
type2_error = np.mean(male_predictions != 1) * 100  
print("Type 2 error (Miss) % male samples misclassified as female:", type2_error, "%")

# Calculate precision and recall for the classifier
true_positives = np.sum(male_predictions == 1) # male classified as male
false_positives = np.sum(female_predictions == 1) # female classified as male
false_negatives = np.sum(male_predictions == -1) # male classified as female
true_negatives = np.sum(female_predictions == -1) #female classified as female

precision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)

print("Precision:", precision)
print("Recall:", recall)


plt.show()
