# SINDy from Scratch
This project implements the Sparse Identification of Nonlinear Dynamical systems (SINDy) algorithm from scratch. The SINDy algorithm is used to identify the underlying dynamical system from time-series data. This project generates synthetic data using the Lorenz system, and apply the SINDy algorithm to identify the governing equations. The project includes the implementation of the Lorenz system data generation, numerical differentiation, and sparse regression using LASSO.

The idea of using the Lorenz system as a benchmark for testing the SINDy algorithm is inspired by the book **"Data-Driven Methods for Dynamic Systems"** by Jason J. Bramburger (2023). This project was developed for educational purposes as part of my learning process to understand and apply the SINDy algorithm.

## Usage
To use this project, first ensure you have the required dependencies installed by running:
```sh
pip install -r requirements.txt
```

You can generate the Lorenz system data and visualize it by running the `synthetic_data.py` script:
```sh
python synthetic_data.py
```

To apply the SINDy algorithm and identify the dynamical system, run the `sindy_model.py` script:
```sh
python sindy_model.py
```
This will fit the SINDy model to the generated data, display the identified coefficients, and plot the original and reconstructed dynamics for comparison.