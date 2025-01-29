import numpy as np
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
from synthetic_data import LorenzSystemData


class SINDy:
    def __init__(self, data, t, sparsity_parameter=0.1, dictionary=None):
        """
        Parameters:
        - data: numpy array of shape (N, 3) containing the states [x, y, z] over time.
        - t: numpy array containing the time vector corresponding to the data.
        - sparsity_parameter: Regularization parameter for sparse regression (LASSO).
        - dictionary: List of candidate functions (callable). If None, defaults to polynomials.
        """
        self.data = data
        self.t = t
        self.sparsity_parameter = sparsity_parameter
        
        # Default dictionary of candidate functions (if no dictionary is provided)
        self.dictionary = dictionary if dictionary is not None else self.default_candidate_functions()

        # Compute the time derivatives using numerical differentiation
        self.ddata = self.compute_derivatives(data, t)

    def compute_derivatives(self, data, t):
        """
        Compute the time derivatives of the states using finite differences.
        """
        dt = t[1] - t[0]  # Assume uniform time step
        dx = np.gradient(data[:, 0], t)  # dx/dt
        dy = np.gradient(data[:, 1], t)  # dy/dt
        dz = np.gradient(data[:, 2], t)  # dz/dt
        return np.column_stack([dx, dy, dz])

    def default_candidate_functions(self):
        """
        Return the default dictionary of candidate functions (polynomials and interactions).
        """
        def f_x(data): return data[:, 0]  # x
        def f_y(data): return data[:, 1]  # y
        def f_z(data): return data[:, 2]  # z
        def f_x2(data): return data[:, 0]**2  # x^2
        def f_y2(data): return data[:, 1]**2  # y^2
        def f_z2(data): return data[:, 2]**2  # z^2
        def f_xy(data): return data[:, 0] * data[:, 1]  # x*y
        def f_xz(data): return data[:, 0] * data[:, 2]  # x*z
        def f_yz(data): return data[:, 1] * data[:, 2]  # y*z
        return [f_x, f_y, f_z, f_x2, f_y2, f_z2, f_xy, f_xz, f_yz]

    def candidate_functions(self, data):
        """
        Apply the candidate functions (from dictionary) to the data.
        """
        library = np.column_stack([func(data) for func in self.dictionary])
        return library

    def fit(self):
        """
        Perform sparse regression to identify the dynamical system using LASSO.
        """
        # Step 1: Build the library of candidate functions for each data point
        library = self.candidate_functions(self.data)
        
        # Step 2: Perform sparse regression using LASSO to find coefficients
        model = Lasso(alpha=self.sparsity_parameter)  # Lasso regression with regularization parameter
        model.fit(library, self.ddata)
        
        # Step 3: Store the identified coefficients (sparse)
        self.coef_ = model.coef_

        # Step 4: Display the nonzero terms and their coefficients
        print("Identified Nonzero Coefficients:")
        non_zero_idx = np.where(self.coef_ != 0)[0]
        terms = [func.__name__ for func in self.dictionary]
        for idx in non_zero_idx:
            print(f"{terms[idx]}: {self.coef_[idx]}")

    def reconstruct_dynamics(self, data):
        """
        Reconstruct the dynamics of the system using the identified sparse model.
        """
        library = self.candidate_functions(data)
        
        # Reconstruct the dynamics based on the nonzero coefficients
        reconstructed_dynamics = np.dot(library, self.coef_.T)
        return reconstructed_dynamics

    def plot_identified_dynamics(self):
        """
        Plot the original and identified dynamics for comparison.
        """
        # Reconstruct the dynamics from the identified model
        reconstructed_dynamics = self.reconstruct_dynamics(self.data)

        # Plot original vs reconstructed for each state (x, y, z)
        plt.figure(figsize=(12, 6))
        
        plt.subplot(3, 1, 1)
        plt.plot(self.t, self.ddata[:, 0], label="Original dx/dt")
        plt.plot(self.t, reconstructed_dynamics[:, 0], label="Reconstructed dx/dt", linestyle="--")
        plt.legend()
        plt.title("x component")

        plt.subplot(3, 1, 2)
        plt.plot(self.t, self.ddata[:, 1], label="Original dy/dt")
        plt.plot(self.t, reconstructed_dynamics[:, 1], label="Reconstructed dy/dt", linestyle="--")
        plt.legend()
        plt.title("y component")

        plt.subplot(3, 1, 3)
        plt.plot(self.t, self.ddata[:, 2], label="Original dz/dt")
        plt.plot(self.t, reconstructed_dynamics[:, 2], label="Reconstructed dz/dt", linestyle="--")
        plt.legend()
        plt.title("z component")

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    lorenz_system = LorenzSystemData()
    
    # Test the SINDy model with default polynomial dictionary
    sindy_model = SINDy(data=lorenz_system.data, t=lorenz_system.t)
    sindy_model.fit()
    sindy_model.plot_identified_dynamics()
    
    # Test the SINDy model with custom dictionary of candidate functions
    dictionary = [
        lambda data: data[:, 0] * data[:, 1],  # x * y
        lambda data: data[:, 1] * data[:, 2],  # y * z
        lambda data: data[:, 2] * data[:, 0],  # z * x
    ]

    sindy_model = SINDy(
        data=lorenz_system.data,
        t=lorenz_system.t,
        sparsity_parameter=0.1,
        dictionary=dictionary
    )

    sindy_model.fit()
    sindy_model.plot_identified_dynamics()

    # Test the SINDy model with different sparsity parameters
    sparsity_params = [2, 0.5, 0.1]

    for sparsity_param in sparsity_params:
        print(f"Sparsity Parameter: {sparsity_param}")
        sindy_model = SINDy(
            data=lorenz_system.data,
            t=lorenz_system.t,
            sparsity_parameter=sparsity_param
        )
        sindy_model.fit()
        sindy_model.plot_identified_dynamics()
        print("\n")
