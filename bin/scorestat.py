'''
Copyright (C) 2020-2021 Mo Zhou <cdluminate@gmail.com>
Released under the Apache-2.0 License.
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
import pandas as pd
import rich
c = rich.get_console()
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = 'Times New Roman'
rcParams['font.size'] = 14
sns.set_theme(style='ticks')

c.print('Fashion-MNIST')
x = np.loadtxt('fashion-scorelog.csv')
mask = np.abs(x[:,1]-1.0) < 1e-3
c.print('number of fully successful attacks:',
        mask.sum(),
        'ratio:',
        mask.sum()/len(x))
c.print('their average query number',
        x[mask, 0].mean())

# https://seaborn.pydata.org/examples/joint_histogram.html
#g = sns.JointGrid(x=x[:,1], y=x[:,0], marginal_ticks=True)
#g.plot_joint(sns.histplot, discrete=(True, True)
#        )
#g.plot_marginals(sns.histplot, element='step')
df = pd.DataFrame(data={'$\\tau_S$': x[:,1], 'Number of Queries': x[:,0]})
plt.figure()
plt.xlabel('Number of Queries')
plt.ylabel('$\\tau_S$')
g = sns.jointplot(data=df, y='$\\tau_S$', x='Number of Queries', kind='hex', marginal_ticks=True)
plt.savefig('fa-joint.svg')


plt.clf()

plt.hist(x[:,1], bins=10, rwidth=0.8, align='right')
plt.grid(True)
plt.title('$\\tau_S$ Histogram on Fashion-MNIST')
plt.xlabel('$\\tau_S$')
plt.ylabel('Number of Data Points')
plt.savefig('fa-score-hist.svg')

plt.clf()


c.print('SOP')
x = np.loadtxt('sop-scorelog.csv')
plt.hist(x[:,1], bins=10, rwidth=0.8, align='right')
plt.grid(True)
plt.title('$\\tau_S$ Histogram on SOP')
plt.xlabel('$\\tau_S$')
plt.ylabel('Number of Data Points')
plt.savefig('sop-score-hist.svg')

plt.clf()

mask = np.abs(x[:,1]-1.0) < 1e-3
c.print('number of fully successful attacks:',
        mask.sum(),
        'ratio:',
        mask.sum()/len(x))
c.print('their average query number',
        x[mask, 0].mean())

df = pd.DataFrame(data={'$\\tau_S$': x[:,1], 'Number of Queries': x[:,0]})
plt.figure()
plt.xlabel('Number of Queries')
plt.ylabel('$\\tau_S$')
g = sns.jointplot(data=df, y='$\\tau_S$', x='Number of Queries', kind='hex', marginal_ticks=True)
plt.savefig('sop-joint.svg')
