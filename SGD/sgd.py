import numpy as np
from scipy.linalg import expm

from matplotlib import pyplot as plt


def calculate_cost(sequence):

    length = len(sequence) 
    time_step = 2*np.pi/length  

    sigma_x = 1/2 * np.mat([[0,1],\
                 [1,0]], dtype=complex)
    sigma_z = 1/2 * np.mat([[1,0],\
                 [0,-1]], dtype=complex)

    evolution_operator = np.matrix(np.identity(2, dtype=complex)) # Initial Evolution operator

    control_strength = 4  # Control field strength
    beta=1
    for action in sequence:
        hamiltonian = action * control_strength * sigma_z + beta*sigma_x # Hamiltonian
        evolution_operator = expm(-1j * hamiltonian * time_step) * evolution_operator  # Evolution operator

    initial_state = np.mat([[1],[0]], dtype=complex) # Initial state
    final_state = evolution_operator * initial_state  # Final state

    target_state = np.mat([[0], [1]], dtype=complex)                             # Target state (south pole)

    error = 1 - (np.abs(final_state.T* target_state)**2).item(0).real  # Infidelity (to minimize)

    return error


step_size = 0.01 #learning rate
global cost_history
cost_history = []

def gradient_descent(sequence, dimension, learning_rate, num_iterations):
    for _ in range(num_iterations):
        random_vector = np.random.rand(dimension) 
        sequence_plus = sequence + random_vector * step_size
        sequence_minus = sequence - random_vector * step_size
        error_derivative = (calculate_cost(sequence_plus) - calculate_cost(sequence_minus)) / (2 * step_size)
        sequence = sequence - (learning_rate) * error_derivative * random_vector
        cost_history.append(calculate_cost(sequence_plus))
        # print(cost_history)
        print(sequence)
    return calculate_cost(sequence)


sequence_length = 20
random_sequence = np.random.rand(sequence_length)
max_epochs = 500
final_fidelity = 1 - gradient_descent(random_sequence, sequence_length, 0.01, max_epochs)

print('Final Fidelity:', final_fidelity)
# print(cost_history)
plt.plot(cost_history)
plt.xlabel("iteration")
plt.ylabel("cost")
# plt.legend()
plt.show()
