import numpy as np
import matplotlib.pyplot as plt

def MMLactivation(x, leak=0.01):
    # this is from LEMBAS authors, 3 components
    # x < 0 is a leaky component, stops gradient getting stuck as zero
    # 0 < x < 0.5 is a linear components - just returns x as is
    # x > 0.5 is curved component 


    x = np.where(x < 0, x * leak, x)
    x = np.where(x > 0.5, 1 - 0.25/x, x) #Pyhton will display division by zero warning since it evaluates both before selecting
    return x


leak = 0.01

x = np.linspace(-2, 3, 1001)
y = MMLactivation(x.copy(), leak)

# x2 = numpy.linspace(0, 1, 10)
# y2 = activationFunctions.MMLactivation(x2.copy(), leak)

plt.rcParams["figure.figsize"] = (3,3)
plt.plot(x, y)
#plt.scatter(x2, y2)

plt.plot([-2, 3], [0, 0], 'black')
plt.ylim([-0.05, 1])
plt.savefig('../figures/MML_visualised')

#plt.gca().set_aspect('equal')