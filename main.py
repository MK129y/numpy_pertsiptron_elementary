import numpy as np

def sigmoid(x):
    return 1/(1+ np.exp(-x))

traning_input = np.array([[0,0,1],
                          [1,1,1],
                          [1,0,1],
                          [0,1,0]])
traning_outputs = np.array([[0,1,1,0]]).T
np.random.seed(1)

synaptic_weights = 2*np.random.random((3,1))- 1
print('Случайные инициализирующие веса:')
print(synaptic_weights)

#Метод обратного распростронения
for i in range(20000):
    input_layer = traning_input
    output = sigmoid(np.dot(input_layer, synaptic_weights))

    err = traning_outputs - output
    adjusttments = np.dot( input_layer.T, err * (output*(1- output)) )

    synaptic_weights += adjusttments

print('Vesa posle obutchenia: ')
print(synaptic_weights)

print('Rezultat posle obutchenia: ')
print(output)

#Тест
new_inputs = np.array([1,1,0])
output = sigmoid(np.dot( new_inputs, synaptic_weights))

print('New situatsia')
print(output)
