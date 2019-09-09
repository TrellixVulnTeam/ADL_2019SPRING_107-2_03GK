import pickle
import matplotlib.pyplot as plt

with open('/home/xiec/ADL/hw3/pkl/step400000.pkl', 'rb') as file1:
    step400000 =pickle.load(file1)
with open('/home/xiec/ADL/hw3/pkl/average_gamma0.85.pkl', 'rb') as file2:
    average_gamma085 =pickle.load(file2)
with open('/home/xiec/ADL/hw3/pkl/average_updatef5000.pkl', 'rb') as file3:
    average_updatef5000 =pickle.load(file3)
with open('/home/xiec/ADL/hw3/pkl/average_newachtecture.pkl', 'rb') as file4:
    New_architecture =pickle.load(file4)
with open('/home/xiec/ADL/hw3/pkl/average_origin.pkl', 'rb') as file0:
    origin = pickle.load(file0)

plt.figure()
plt.title('Learning Curve')
x =  step400000[:101]
y0 = origin[:101]
y1 = average_gamma085[:101]
y2 = average_updatef5000[:101]
y3 = New_architecture[:101]
print(x)
print(len(x),len(y3))
plt.plot(x, y0, color='green', label='Origin')
plt.plot(x, y1, color='red', label='Gamma0.85')
plt.plot(x, y2,  color='yellow', label='Update_freq5000')
plt.plot(x, y3, color='blue', label='New_architecture')

plt.xlabel('Steps')
plt.ylabel('Avg reward')
plt.show()