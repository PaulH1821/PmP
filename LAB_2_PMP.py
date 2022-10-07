import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import arviz as az

np.random.seed(1)

x=
mec1 = stats.expon.rvs(scale=1/4, size=10000)
mec2 = stats.expon.rvs(scale=1/6,size=10000)
z = 40/100*mec1 +






az.plot_posterior({'mec1':mec1,'mec2':mec2,'z':z})

plt.show()


