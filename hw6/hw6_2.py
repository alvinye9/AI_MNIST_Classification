import numpy as np
import matplotlib.pyplot as plt


#================ Question 2 Part a ================
num_coins = 1000
flips_per_coin = 10
experiments = 100000

V1_values = np.zeros(experiments)
Vrand_values = np.zeros(experiments)
Vmin_values = np.zeros(experiments)

# repeat  for 100000 runs
for i in range(experiments):
    # Simulate flipping 1000 coins 10 times
    flips = np.random.randint(2, size=(num_coins, flips_per_coin))
    
    # Calculate V1, mean of all 10 flips of the first coin
    # print("Number of Flips for coin 1: ",flips[0, :].shape )
    V1_values[i] = flips[0, :].mean()
    
    # calculate Vrand, mean of all 10 flips of a random coin
    rand_coin_index = np.random.randint(num_coins)
    Vrand_values[i] = flips[rand_coin_index, :].mean()
    
    # calculate Vmin, minimum mean 
    min_coin_index = np.argmin(np.mean(flips, axis=1)) #returns index of minimum mean (one of the 1000 coins)
    Vmin_values[i] = flips[min_coin_index, :].mean()


plt.figure(1, figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.hist(V1_values, bins=30, edgecolor='k', alpha=0.7)
plt.title('Distribution of V_{1}')
plt.xlabel('Fraction of Heads')
plt.ylabel('Frequency')

plt.subplot(1, 3, 2)
plt.hist(Vrand_values, bins=30, edgecolor='k', alpha=0.7)
plt.title('Distribution of V_{rand}')
plt.xlabel('Fraction of Heads')

plt.subplot(1, 3, 3)
plt.hist(Vmin_values, bins=30, edgecolor='k', alpha=0.7)
plt.title('Distribution of V_{min}')
plt.xlabel('Fraction of Heads')

plt.tight_layout()
plt.savefig("2b_histograms.jpg")



#================ Question 2 Part b ================
# Define μ1, μrand, and μmin
mu1 = 0.5
mu_rand = 0.5  
mu_min = 0 # minimum possible fraction of heads

# Calculate Hoeffding's bound for different values of ϵ
epsilons = np.linspace(0, 0.5, num=11) #values from 0 to 0.5 increments of 0.05
hoeffding_bound = 2 * np.exp(-2 * (epsilons**2) * flips_per_coin)

prob_v1 = np.zeros(len(epsilons))
prob_vrand = np.zeros(len(epsilons))
prob_vmin = np.zeros(len(epsilons))

# Calculate estimated probabilities for different values of ϵ
# P(|V - μ| > ϵ)
for i, epsilon in enumerate(epsilons):
    count_v1 = np.sum(np.abs(V1_values - mu1) > epsilon) # Count of instances where |V - μ| > ϵ
    prob_v1[i] = count_v1 / experiments
    print("P_V1: ", prob_v1[i])
    
    count_vrand = np.sum(np.abs(Vrand_values - mu_rand) > epsilon)
    prob_vrand[i] = count_vrand / experiments
    print("P_Vrand: ", prob_vrand[i])
    
    count_vmin = np.sum(np.abs(Vmin_values - mu_min) > epsilon)
    prob_vmin[i] = count_vmin / experiments
    print("P_Vmin: ", prob_vmin[i])

plt.figure(2, figsize=(8, 6))

plt.plot(epsilons, prob_v1, label='$P(|V_1 - \mu_1| > \epsilon)$', linewidth=5)
plt.plot(epsilons, prob_vrand, label='$P(|V_{rand} - \mu_{rand}| > \epsilon)$')
plt.plot(epsilons, prob_vmin, label='$P(|V_{min} - \mu_{min}| > \epsilon)$')
plt.plot(epsilons, hoeffding_bound, label="Hoeffding's Bound", linestyle='--')

plt.xlabel('$\epsilon$')
plt.ylabel('Probability')
plt.title('Estimated Probabilities and Hoeffding\'s Bound')
plt.legend()
plt.grid(True)
plt.savefig("2c_Hoeffding_Bound")
plt.show()

