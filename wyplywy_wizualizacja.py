import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np

xt500  = pd.read_csv('results_x_t_500.csv', header = None).values 
xt1000 = pd.read_csv('results_x_t_1000.csv', header = None).values 
xt2000 = pd.read_csv('results_x_t_2000.csv').values 
xt5000 = pd.read_csv('results_x_t_5000.csv').values 

for i in [50, 500, 1000, 5000]:
    df = pd.read_csv('results_x_t_' + str(i) + '.csv', header = None).values
    print(i, ': ', ' max: ', max(abs(df[:, 8])), ' min: ', min(abs(df[:, 8]))) 
    
for i in [50, 500, 5000]:
    df = pd.read_csv('results_x_t_' + str(i) + '.csv', header = None).values
    print(df[:, 0].shape[0])
    print(i, ': ', 'L2 norm:', np.sqrt((df[:, 0] + 50 * np.sin(np.arange(0, 5, 5.0/i) * np.pi) + 100)**2)) 
    plt.plot(np.arange(0, 5, 5.0/i), np.sqrt((df[:, 0] + 50 * np.sin(np.arange(0, 5, 5.0/i) * np.pi) + 100)**2), label = '$\Delta t =$' + str(5.0/i))
    

plt.xlabel('czas')
plt.ylabel('błąd w normie $L^2$')
plt.legend()
plt.grid()
plt.show()
    
x = np.arange(0, 5, 0.01) 
y1 = 50 * np.sin(x * np.pi) 
y2 = 50 - y1 
y3 = 50 + y1 


fig = plt.figure(1) 

ax1 = fig.add_subplot(311)
ax1.plot(x, y1)
ax1.grid()
ax1.set_ylabel('$Q_2$')
ax1.set_title('Rozkład wielkości wydatków\nna brzegach (2), (3) i (4)', weight = 'bold')

ax2 = fig.add_subplot(312)
ax2.plot(x, y2)
ax2.grid()
ax2.set_ylabel('$Q_3$')

ax3 = fig.add_subplot(313)
ax3.plot(x, y3)
ax3.grid()
ax3.set_ylabel('$Q_4$')
ax3.set_xlabel('czas symulacji')

plt.subplots_adjust(hspace = 0.5)

plt.savefig('rozklad_Q.pdf')
plt.show()




#M1s   = np.zeros((6, 4)) 
#M1s[0, :] = xt500[99, 0:4]
#M1s[1, :] = xt1000[199, 0:4]
#M1s[2, :] = xt2000[399, 0:4]
#M1s[3, :] = xt3000[599, 0:4]
#M1s[4, :] = xt4000[799, 0:4]
#M1s[5, :] = xt5000[999, 0:4]


#M2_5s = np.zeros((6, 4))
#M2_5s[0, :] = xt500[249, 0:4]
#M2_5s[1, :] = xt1000[499, 0:4]
#M2_5s[2, :] = xt2000[999, 0:4]
#M2_5s[3, :] = xt3000[1499, 0:4]
#M2_5s[4, :] = xt4000[1999, 0:4]
#M2_5s[5, :] = xt5000[2499, 0:4]

#plt.figure(1)
#plt.plot(xt500[:, 0])

##plt.figure(2)
#plt.plot(xt2000[:, 0][::4][:-1])
 
      
#plt.figure(3) 
#plt.plot(np.abs(xt2000[:, 0][::4][:-1] - xt500[:, 0])) 
#plt.show()

#print('maximum error: ', np.max(np.abs(xt2000[:, 0][::4][:-1] - xt500[:, 0])))



#M4s   = np.zeros(6, 4) 

#M5s   = np.zeros(6, 4)
