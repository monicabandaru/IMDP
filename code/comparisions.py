import matplotlib.pyplot as plt
import numpy as np

k=[50,100,150,200]
#k1,k2,k3,k4=-85,-75,-83,-70
b1=[8870,8895,8999,9050]
b2=[8155,8560,8825,8858]
b3=[7140,7186,7496,7913]
b4=[6250,6280,6300,6394]
b5=[6170,6019,6181,5955]

fig=plt.figure()
fig.show()
ax=fig.add_subplot(111)

ax.plot(k,b1,c='b',label='Greedy',fillstyle='none')
ax.plot(k,b2,c='g',ls='--',label='Celf')
ax.plot(k,b3,c='k',ls='-',label='Celf++')
ax.plot(k,b4,c='r',ls='-',label='IMM')
ax.plot(k,b5,c='m',ls='--',label='IM_DP',fillstyle='none')
#ax.plot(x,x-1,c='k',marker="+",ls=':',label='MSD')
plt.xlabel('SeedSet Size', fontweight='bold', fontsize=10)
plt.ylabel('Influence Spread', fontweight='bold', fontsize=10)
plt.legend(loc=2)
plt.draw()
plt.show()