import numpy as np
from scipy.stats import truncnorm

def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


"PART 1: Doing MC as a reference base"
n = 5000
X_samples = np.random.normal(0, 1, n)
i = 0
for sample in X_samples: # Applying f(X) to each randomly drawn x
    sample = sample**2
    X_samples[i] = sample
    i += 1
MC = (X_samples.sum())/n # Print MC Results
print "MC results: " + str(MC)


"PART 2: Importance Sampling"
# Creating an array filled of U distribution samples from interval [-5, 5]
U_normal = get_truncated_normal(mean=-5, sd=5, low=-5, upp=5)
U_samples = []
counter = 0
while counter < n:
    u_sample = U_normal.rvs()
    U_samples.append(u_sample)
    counter += 1
U_samples = np.asarray(U_samples)
# Evaluating samples by probability distribution
i = 0
for sample in U_samples:
    sample = (1/np.sqrt(2*np.pi*(5**2)))*np.exp(-((sample+5) ** 2) / (2 * (5 ** 2)))
    U_samples[i] = sample
    i += 1
i = 0
X_samples = np.random.normal(0, 1, n)
for sample in X_samples:
    sample = (1 / np.sqrt(2 * np.pi * (1 ** 2))) * np.exp(-((sample-0) ** 2) / (2 * (1 ** 2)))
    X_samples[i] = sample
    i += 1
# Rest of setup to run calculations
counter = 0
calculations = []
# Calculations
while(counter < n):
    calc = ((U_samples[counter])**2)*((X_samples[counter])/(U_samples[counter]))
    calculations.append(calc)
    counter += 1
calculations = np.asarray(calculations) # Convert into np array
IS = (calculations.sum())/n
print "Important Sampling results: " + str(IS)




