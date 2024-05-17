import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# Read the training data
train_cat = np.matrix(np.loadtxt('train_cat.txt', delimiter=',')) #[64 x K1], K1 blocks of pixels, 64 pixels per block (each block is 8x8 pixels)
train_grass = np.matrix(np.loadtxt('train_grass.txt', delimiter=',')) #[64 x K0]
# print(np.shape(train_cat)) 
# print(np.shape(train_grass))

# Estimate means
mu1 = np.mean(train_cat, axis=1) #[64 x 1]
mu0 = np.mean(train_grass, axis=1) #[64 x 1]
# print(np.shape(mu1))
# print(np.shape(mu0))

# Estimate covariance matrices
Sigma1 = np.cov(train_cat) #[64 x 64]
Sigma0 = np.cov(train_grass) #[64 x 64]
# print(np.shape(Sigma1))
# print(np.shape(Sigma0))

# find inverses of covariance matrices to make future calcs faster
Sigma_inv1 = np.linalg.inv(Sigma1)
Sigma_inv0 = np.linalg.inv(Sigma0)

# find log of det of covariance matrices to make future calcs even faster
log_det_Sigma1 = np.log(np.linalg.det(Sigma1))
log_det_Sigma0 = np.log(np.linalg.det(Sigma0))

# estimate priors (pi)
K1 = train_cat.shape[1]
K0 = train_grass.shape[1]
pi1 = K1 / (K1 + K0) 
pi0 = K0 / (K1 + K0)


# log(posterior) calculated using multi-dimensional gaussian liklihood
def compute_log_posterior(block):
    # Reshape block to a column vector [64 x 1]
    block = block.reshape(-1, 1)

    log_posterior_cat = -0.5 * (block - mu1).T * Sigma_inv1 * (block - mu1) + np.log(pi1) - 0.5 * log_det_Sigma1
    log_posterior_grass = -0.5 * (block - mu0).T * Sigma_inv0 * (block - mu0) + np.log(pi0) - 0.5 * log_det_Sigma0
    # print(log_posterior_cat)
    # print(log_posterior_grass)
    return log_posterior_cat, log_posterior_grass



# ============================ HW3 Exercise 3, Part bc ============================
def compute_TP_FP(predictions, ground_truth):
    # print(np.shape(predictions))
    # print(np.shape(ground_truth))
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
# print(M)
# print(N)

# Read ground truth image, ignore boundary pixels
truth_uncropped = np.array(Image.open('truth.png'))
truth = truth_uncropped[0:M-8, 0:N-8]
# print(np.shape(truth_uncropped))
# print(np.shape(truth))

# Initialize predicted binary image
prediction = np.zeros((M-8, N-8))

thresholds = np.linspace(-2, 2, 20)
pD_values = np.zeros_like(thresholds)
pF_values = np.zeros_like(thresholds)

for index, tau in enumerate(thresholds):
    for i in range(M-8):
        for j in range(N-8):
            # Extract 8x8 block
            block = Y[i:i+8, j:j+8]

            # Compute log posterior probabilities (scalar values)
            log_posterior_cat, log_posterior_grass = compute_log_posterior(block)
            ratio = log_posterior_cat / log_posterior_grass
            # ratio_scaled = ratio * (pi1/pi0)
            # print(ratio)
            # print(ratio_scaled)
            # Assign pixel to the class with the highest log posterior probability
            if  ratio > tau:
                prediction[i, j] = 1  # Class cat
            else:
                prediction[i, j] = 0  # Class grass

    plt.figure(index)
    plt.imshow(prediction, cmap='gray')
    # plt.show()

    # Compute TP and FP
    TP, FP = compute_TP_FP(prediction, truth)
    total_positives = np.sum(truth >= 0.5)
    total_negatives = np.sum(truth < 0.5)
    pD = TP / total_positives   # Compute pD and pF
    pF = FP / total_negatives
    print("trying ", index, "th threshold: ", tau)
    print("True Positives: ", TP)
    print("False Positives: ", FP)
    print("Total positives in ground truth: ", total_positives)
    print("Total negatives in ground truth: ", total_negatives)
    print("Prob of Detection: ", pD ) 
    print("Prob of False Alarm: ", pF)
 
    pD_values[index] = pD
    pF_values[index] = pF

# test classification using operating point (tau = pi1/pi0)
for i in range(M-8):
    for j in range(N-8):
        # Extract 8x8 block
        block = Y[i:i+8, j:j+8]

        # Compute log posterior probabilities (scalar values)
        log_posterior_cat, log_posterior_grass = compute_log_posterior(block)
        ratio = log_posterior_cat / log_posterior_grass
        # print(ratio)
        # print(ratio_scaled)
        # Assign pixel to the class with the highest log posterior probability
        if  ratio > pi1/pi0:
            prediction[i, j] = 1  # Class cat
        else:
            prediction[i, j] = 0  # Class grass

plt.figure(index)
plt.imshow(prediction, cmap='gray')

# Compute TP and FP
TP, FP = compute_TP_FP(prediction, truth)
total_positives = np.sum(truth >= 0.5)
total_negatives = np.sum(truth < 0.5)
pD = TP / total_positives   # Compute pD and pF
pF = FP / total_negatives
print("trying operating point: ", pi1/pi0)
print("True Positives: ", TP)
print("False Positives: ", FP)
print("Total positives in ground truth: ", total_positives)
print("Total negatives in ground truth: ", total_negatives)
print("Prob of Detection: ", pD ) 
print("Prob of False Alarm: ", pF)


# pD_values = np.append(pD_values, pD)
# pF_values = np.append(pF_values, pF)
pD_operating = pD
pF_operating = pF

# Plot the ROC curve
plt.figure(99)
plt.plot(pF_values, pD_values)
plt.plot(pF_operating, pD_operating, 'ro')
plt.xlabel('Probability of False Alarm (pF)')
plt.ylabel('Probability of Detection (pD)')
plt.title('ROC Curve')
plt.grid(True)
plt.savefig("3bc_ROC.png")

plt.show()