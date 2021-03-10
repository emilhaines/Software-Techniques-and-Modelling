#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

N = 10000 # number MC experiments
n = 15 # experimental data sample size
mu = 1 # true mean
var = 2 # true variance
np.random.seed(42)
data = np.sqrt(var)*np.random.normal(size=(N, n)) + mu
#%%
fig, axs = plt.subplots(4, figsize=(5, 18))
fig.suptitle("Examples of experimental samples")
for i in range(4):
    axs[i].hist(data[i, :], histtype='step')
plt.show()
# %%
# Compare mean with median

# MEAN
T1 = np.mean(data, axis=1)
T1m = np.mean(T1) # estimator
b1 = T1m - mu # bias
s21 = np.var(T1)
counts1,_,_ = plt.hist(T1, histtype='step')
plt.plot([T1m, T1m], [0, np.max(counts1)], 'k--' ,label='Estimator')
plt.title('Distribution of sample means')
plt.legend()
plt.show()
# %%
# MEDIAN
T2 = np.median(data, axis=1)
T2m = np.mean(T1) # estimator
b2 = T1m - mu # bias
s22 = np.var(T2)
counts2,_,_ = plt.hist(T2, histtype='step')
plt.plot([T2m, T2m], [0, np.max(counts2)], 'k--' ,label='Estimator')
plt.title('Distribution of sample medians')
plt.legend()
plt.show()
# %%
print("Relative efficiency = {}".format(s21/s22))
# %%
# Compare standard deviation and MAD estimator

# STD
std = np.std(data, axis=1)
stdm = np.mean(std) # estimator
stdv = np.var(std)
counts3,_,_ = plt.hist(std, histtype='step')
plt.plot([stdm, stdm], [0, np.max(counts3)], 'k--', label='Estimator')
plt.title('Distribution of sample std')
plt.legend()
plt.show()
# %%
# MAD
mad = stats.median_absolute_deviation(data, axis=1)
madm = np.mean(mad) # estimator
madv = np.var(mad)
counts4,_,_ = plt.hist(mad, histtype='step')
plt.plot([madm, madm], [0, np.max(counts4)], 'k--', label='Estimator')
plt.title('Distribution of sample MAD')
plt.legend()
plt.show()
# %%
print("Relative efficiency = {}".format(stdv/madv))
# %%
