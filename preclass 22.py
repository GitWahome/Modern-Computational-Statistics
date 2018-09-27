from scipy.stats import gamma, norm, beta
import matplotlib.pyplot as plt
import numpy as np

#followed https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.gamma.html
#Figure configurations
fig, ax = plt.subplots(1, 1)

#Gamma configs
a = 1.99323054838
mean, var, skew, kurt = gamma.stats(a, moments='mvsk')
x = np.linspace(gamma.ppf(0.01, a), gamma.ppf(0.99, a), 100)
ax.plot(x, gamma.pdf(x, a),'r-', lw=5, alpha=0.6, label='gamma pdf')
rv = gamma(a)
ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')
vals = gamma.ppf([0.001, 0.5, 0.999], a)
np.allclose([0.001, 0.5, 0.999], gamma.cdf(vals, a))
#Generate random values
r = gamma.rvs(a, size=1000)
ax.hist(r, normed=True, histtype='stepfilled', alpha=0.2)
ax.legend(loc='best', frameon=False)

#Normal configs
normVals = norm.rvs(a, size = 100)
fit = norm.pdf(normVals, np.mean(normVals), np.std(normVals))
plt.plot(normVals,fit,'-o')
plt.hist(normVals,normed=True)
#Beta configs
plt.show()