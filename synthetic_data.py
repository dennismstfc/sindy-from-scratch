import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

class LorenzSystemData:
    def __init__(
            self, 
            sigma=10.0, 
            rho=28.0, 
            beta=8/3, 
            initial_conditions=[1.0, 0.0, 0.0], 
            t_span=(0, 100), 
            num_points=20000
            ):
        # Parameters for the Lorenz system
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        
        self.initial_conditions = initial_conditions
        
        self.t_span = t_span
        self.num_points = num_points
        
        self.t = np.linspace(self.t_span[0], self.t_span[1], self.num_points)
        
        self.data = self.solve_lorenz_system()

    def lorenz_system(self, state, t):
        x, y, z = state
        
        dxdt = self.sigma * (y - x)
        dydt = x * (self.rho - z) - y
        dzdt = x * y - self.beta * z
        
        return [dxdt, dydt, dzdt]
    
    def solve_lorenz_system(self):
        solution = odeint(self.lorenz_system, self.initial_conditions, self.t)
        return solution

    def plot_data(self):
        fig, axs = plt.subplots(3, 1, figsize=(10, 18))
        
        # Plot y vs x
        axs[0].plot(self.data[:, 0], self.data[:, 1], color='b')
        axs[0].set_title('y vs x')
        axs[0].set_xlabel('x')
        axs[0].set_ylabel('y')
        
        # Plot z vs y
        axs[1].plot(self.data[:, 1], self.data[:, 2], color='r')
        axs[1].set_title('z vs y')
        axs[1].set_xlabel('y')
        axs[1].set_ylabel('z')
        
        # Plot z vs x
        axs[2].plot(self.data[:, 0], self.data[:, 2], color='g')
        axs[2].set_title('z vs x')
        axs[2].set_xlabel('x')
        axs[2].set_ylabel('z')
        
        fig.suptitle('Lorenz System Attractor', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()


if __name__ == '__main__':
    lorenz_data = LorenzSystemData()
    lorenz_data.plot_data()

    X = lorenz_data.data  # State data (x, y, z) over time
    print(X[:5])  