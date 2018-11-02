import numpy as np
from scipy.stats import truncnorm
import scipy.stats as st

class my_pdf(st.rv_continuous):
    def _pdf(self,x):
        return (1+np.cos(x*np.pi))/2  # Normalized over its range, in this case [-1,1]

"Generating plot of new probabiltiy density function"


"MC for baseline comparision"
X_distribution = my_pdf(a=-1, b=1, name='my_pdf')
i = 0
n = 500
X_samples = []
while i < n:
    sample = X_distribution.rvs()
    X_samples.append(sample)
    i += 1
X_samples = np.asarray(X_samples)
i = 0
for sample in X_samples:
    eval_sample = sample ** 2
    X_samples[i] = eval_sample
    i += 1
MC = (X_samples.sum())/n # Print MC Results
print "MC results: " + str(MC)


"Importance Sampling"
# Creating an array filled of U distribution samples from interval [-1, 1]
U_uniform = np.random.uniform(-5, 5, n)
# Evaluating sample for U for p(Xi)
X_PDF = []
for sample in U_uniform:
    sample = (1 + np.cos(np.pi*sample))/2
    X_PDF.append(sample)
X_PDF = np.asarray(X_PDF)
# q(Xi) is always 1/10th
# set up rest of calculations
i = 0
calculations = []
# Calculations
while(i < n):
    calc = ((U_uniform[i]) ** 2) * ((X_PDF[i]) / 0.1)
    calculations.append(calc)
    i += 1
calculations = np.asarray(calculations) # Convert into np array
IS = (calculations.sum())/n
print "Important Sampling results: " + str(IS)