import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Exercise 1 Part a

# N(mu,sigma), mean, std
mu = 0
sigma = 1

# Define the PDF function
def normal_pdf(x, mu, sigma):
    return (1 / (np.sqrt(2 * np.pi * sigma**2))) * np.exp(-(x - mu)**2 / (2 * sigma**2))

# Generate x values in the range [-3, 3]
x = np.linspace(-3, 3, 1000)

# Calculate the PDF values
pdf_values = normal_pdf(x, mu, sigma)

# Plot the PDF
plt.figure(1)
plt.plot(x, pdf_values, label=r'$f_X(x) = \frac{1}{\sqrt{2\pi}\sigma^2}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.title('Normal Distribution PDF, $\mu={}, \sigma={}$'.format(mu, sigma))
plt.legend()
plt.savefig('1a_normal_distribution_pdf.png')


# Exercise 1 Part b

# 1000 random samples from N(0, 1), 1D Normal Distribution
random_samples = np.random.normal(mu, sigma, 1000)

# Plot the histogram with 4 bins
plt.figure(2,figsize=(12, 6))  # figure size
plt.subplot(1, 2, 1)  # Subplot 1
#histogram of probability density
plt.hist(random_samples, bins=4, density=True,  color='blue', label='Random Samples (4 bins)')
# Add the true PDF curve for comparison
plt.plot(x, pdf_values, color='red', label=r'$f_X(x)$')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.title('Histogram with 4 Bins')
plt.legend()

# Plot the histogram with 1000 bins
plt.subplot(1, 2, 2)  # Subplot 2
plt.hist(random_samples, bins=1000, density=True,  color='green', label='Random Samples (1000 bins)')
# Add the true PDF curve for comparison
plt.plot(x, pdf_values, color='red', label=r'$f_X(x)$')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.title('Histogram with 1000 Bins')
plt.legend()
# Save both histograms in one image
plt.savefig('1b_histograms.png')

# Use scipy.stats.norm.fit to estimate mean and standard deviation
estimated_mu, estimated_sigma = norm.fit(random_samples)
print("Estimated Mean:", estimated_mu)
print("Estimated Standard Deviation:", estimated_sigma)



# Exercise 1 Part c

# Array of risk values
J_hat = []

# Iterate over different bin values (m)
for m in range(1, 201):
    h = (max(random_samples) - min(random_samples)) / m
    n = len(random_samples)

    # np.histogram()[0] returns array whose elements are the number of sample values in each bin
    p_j = np.histogram(random_samples, bins=m)[0] / n

    # Calculate risk J(h)_hat acc to the given formula
    risk = ( 2 / (h * (n - 1)) ) - ( (n + 1) / (h * (n - 1)) ) * np.sum(p_j**2)
    
    J_hat.append(risk)


# Plot J(h)_hat WRT to m for m = [1, 200]
plt.figure(3)
plt.plot(range(1, 201), J_hat, label='Risk Function')
plt.xlabel('Number of Bins (m)')
plt.ylabel('Risk (J(h))')
plt.title('Cross Validation Estimator of Risk')
plt.legend()
plt.savefig('1c_risk_function.png')

optimal_m = np.argmin(J_hat) + 1  # +1 because indexing starts from 1
print("optimal number of bins", optimal_m)

# Plot histogram with optimal m∗
plt.figure(4)
plt.hist(random_samples, bins=optimal_m, density=True, color='purple', label='Random Samples (Optimal Bins)')
# Add the true PDF curve for comparison
plt.plot(x, pdf_values, color='red', label=r'$f_X(x)$')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.title(f'Histogram with Optimal Bins (m={optimal_m})')
plt.legend()
plt.savefig('1c_histogram_with_m*.png')

plt.show() #show all figures