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

print("(2b i) The first 2 entries of the vector μ₁:", mu1[:2])
print("The first 2 entries of the vector μ₀:", mu0[:2])
print("(2b ii) The first 2x2 entries of the matrix Σ₁:")
print(Sigma1[:2, :2])
print("The first 2x2 entries of the matrix Σ₀:")
print(Sigma0[:2, :2])
print("(2b iii) The value of π₁:", pi1)
print("The value of π₀:", pi0)


def compute_log_posterior(block):
    # Reshape block to a column vector [64 x 1]
    block = block.reshape(-1, 1)

    log_posterior_cat = -0.5 * (block - mu1).T * Sigma_inv1 * (block - mu1) + np.log(pi1) - 0.5 * log_det_Sigma1
    log_posterior_grass = -0.5 * (block - mu0).T * Sigma_inv0 * (block - mu0) + np.log(pi0) - 0.5 * log_det_Sigma0
    
    return log_posterior_cat, log_posterior_grass

# ============================ HW3 Exercise 2, Part e ============================

# Read the testing image, remove the excess 3rd dimension
Y = plt.imread('cat_grass_1_compressed.jpg')/ 255
Y = Y[:, :, 0]
# print(np.shape(Y))
M = Y.shape[0]
N = Y.shape[1]
# print(M)
# print(N)

# Initialize predicted binary image
prediction = np.zeros((M-8, N-8))

# Loop through all pixels of the testing image
for i in range(M-8):
    for j in range(N-8):
        # Extract 8x8 block
        block = Y[i:i+8, j:j+8]

        # Compute log posterior probabilities
        log_posterior_cat, log_posterior_grass = compute_log_posterior(block)
       # print("iteration: ", i, " ", j)
        # Assign pixel to the class with the highest log posterior probability
        if log_posterior_cat > log_posterior_grass:
            prediction[i, j] = 1  # Class cat
        else:
            prediction[i, j] = 0  # Class grass
print("done generating prediction binary image")

# Display predicted binary image
plt.figure(1)
plt.imshow(prediction, cmap='gray')
plt.savefig("2e_binary_image.png")



plt.show()