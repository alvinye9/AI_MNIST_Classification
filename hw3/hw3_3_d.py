import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cvxpy as cp


# Read the training data
train_cat = np.matrix(np.loadtxt('train_cat.txt', delimiter=',')) #[64 x K1], K1 blocks of pixels, 64 pixels per block (each block is 8x8 pixels)
train_grass = np.matrix(np.loadtxt('train_grass.txt', delimiter=',')) #[64 x K0]
mu1 = np.mean(train_cat, axis=1) #[64 x 1]
mu0 = np.mean(train_grass, axis=1) #[64 x 1]
Sigma1 = np.cov(train_cat) #[64 x 64]
Sigma0 = np.cov(train_grass) #[64 x 64]
Sigma_inv1 = np.linalg.inv(Sigma1)
Sigma_inv0 = np.linalg.inv(Sigma0)
log_det_Sigma1 = np.log(np.linalg.det(Sigma1))
log_det_Sigma0 = np.log(np.linalg.det(Sigma0))

K1 = train_cat.shape[1]
K0 = train_grass.shape[1]
pi1 = K1 / (K1 + K0) 
pi0 = K0 / (K1 + K0)

def compute_log_posterior(block):
    block = block.reshape(-1, 1)
    log_posterior_cat = -0.5 * (block - mu1).T * Sigma_inv1 * (block - mu1) + np.log(pi1) - 0.5 * log_det_Sigma1
    log_posterior_grass = -0.5 * (block - mu0).T * Sigma_inv0 * (block - mu0) + np.log(pi0) - 0.5 * log_det_Sigma0
    return log_posterior_cat, log_posterior_grass

def compute_TP_FP(predictions, ground_truth):
    TP = 0
    FP = 0
    for i in range(M-8):
        for j in range(N-8):
            if(predictions[i,j] == 1 and ground_truth[i,j] >= 0.5): #doesnt perfectly code as 1 or 0
                TP = TP + 1
            if(predictions[i,j] == 1 and ground_truth[i,j] < 0.5):
                FP = FP + 1
    return TP, FP

# Read the testing image
Y = plt.imread('cat_grass.jpg') / 255
M = Y.shape[0]
N = Y.shape[1]


# Read ground truth image, ignore boundary pixels
truth_uncropped = np.array(Image.open('truth.png'))
truth = truth_uncropped[0:M-8, 0:N-8]


# Constructing the feature matrix A
X1 = train_cat.T  # Transpose to have samples as rows
X0 = train_grass.T
A = np.vstack((X1, X0))  # Stack vertically to form the feature matrix

# Constructing the target vector b
b1 = np.ones((X1.shape[0], 1))  # For class 1
b0 = -1 * np.ones((X0.shape[0], 1))  # For class 0
b = np.vstack((b1, b0))  # Stack vertically to form the target vector

# Solving the regression problem
theta = np.linalg.lstsq(A, b, rcond=None)[0]

# Testing the classifier and plotting the ROC curve
thresholds = np.linspace(-1, 0, 20)
# thresholds = np.linspace(-1, 1, 20)
pD_values = []
pF_values = []


# Initialize predicted binary image
prediction_image = np.zeros((M-8, N-8))

for index, tau in enumerate(thresholds):
    TP = 0
    FP = 0
    for i in range(M-8):
        for j in range(N-8):
            block = Y[i:i+8, j:j+8].reshape(-1, 1)
            prediction = np.dot(theta.T, block)

            if  prediction >= tau:
                prediction_image[i, j] = 1  # Class cat
            else:
                prediction_image[i, j] = 0  # Class grass

            if prediction >= tau and truth[i, j] >= 0.5:
                TP += 1
            elif prediction >= tau and truth[i, j] < 0.5:
                FP += 1

    plt.figure(index)
    plt.imshow(prediction_image, cmap='gray')

    pD = TP / np.sum(truth >= 0.5)
    pF = FP / np.sum(truth < 0.5)
    pD_values.append(pD)
    pF_values.append(pF)

    total_positives = np.sum(truth >= 0.5)
    total_negatives = np.sum(truth < 0.5)

    print("trying ", index, "th threshold: ", tau)
    # print("True Positives: ", TP)
    # print("False Positives: ", FP)
    # print("Total positives in ground truth: ", total_positives)
    # print("Total negatives in ground truth: ", total_negatives)
    # print("prediction: ", prediction)
    print("Prob of Detection: ", pD ) 
    print("Prob of False Alarm: ", pF)


plt.figure(69)
plt.plot(pF_values, pD_values)
plt.xlabel('Probability of False Alarm (pF))')
plt.ylabel('Probability of Detection (pD)')
plt.title('ROC Curve')
plt.grid(True)
plt.savefig("3d_LinReg_ROC.png")
plt.show()
