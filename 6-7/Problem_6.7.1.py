import numpy as np
from scipy.stats import truncnorm

def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


"PART 1: Doing MC as a reference base"
n = 7000
X_samples = np.random.normal(0, 1, n)
eval_X = []
for sample in X_samples: # Applying f(X) to each randomly drawn x
    sample = sample**2
    eval_X.append(sample)
eval_X = np.asarray(eval_X)
MC = (eval_X.sum())/n # Print MC Results
print "MC results: " + str(MC)


"PART 2: Importance Sampling"
# Creating an array of samples from normal distribution
U_uniform = np.random.uniform(-5, 5, n)
# Evaluation PDF of X_samples
i = 0
X_PDF = []
for sample in U_uniform:
    sample = (1 / np.sqrt(2 * np.pi * (1 ** 2))) * np.exp(-(((sample-0) ** 2) / (2 * (1 ** 2))))
    X_PDF.append(sample)
X_PDF = np.asarray(X_PDF)
# Rest of setup to run calculations
i = 0
calculations = []
# Calculations
while(i < n):
    calc = ((U_uniform[i])**2)*((X_PDF[i])/0.1)
    calculations.append(calc)
    i += 1
calculations = np.asarray(calculations) # Convert into np array
IS = (calculations.sum())/n
print "Important Sampling results: " + str(IS)







