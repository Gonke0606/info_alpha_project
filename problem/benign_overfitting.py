import numpy as np
import matplotlib.pyplot as plt

NUM_DATA = 200
NOISE_LEVEL = 0.3

p_values = [50, 100, 250, 500, 1000, 10000]

def generate_cos_vector(x, p):
    if x.ndim == 1:
        x = x[:, np.newaxis]
    k = np.arange(1, p + 1)
    val = np.cos(x * k)
    return val

def generate_weighted_cos_vector(x, p):
    if x.ndim == 1:
        x = x[:, np.newaxis]
    k = np.arange(1, p + 1)

    # 低周波成分ほど影響を大きくするために、各項をkで割る
    val = np.cos(x * k) / k
    return val

def generate_mean_zero_vector(num_data):
    x = np.random.randn(num_data)
    return x - np.mean(x)

def show_graph(p, x_train, y_train, x_plot, y_true, y_pred):
    plt.figure(figsize=(10, 7))
    plt.scatter(x_train, y_train, s=30, label="data", color='blue')
    plt.plot(x_plot, y_true, 'k-', lw=3, label="cos(3x)")
    plt.plot(x_plot, y_pred, lw=2, label=f"pred (p={p})", color='blue')
    plt.title("benign overgrowth", fontsize=16)
    plt.xlabel("x", fontsize=12)
    plt.ylabel("y", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--')
    plt.ylim(-2, 2)
    plt.show()

x_train = np.pi * np.random.rand(NUM_DATA)
noise = generate_mean_zero_vector(NUM_DATA) * NOISE_LEVEL
y_train = np.cos(3 * x_train) + noise

x_plot = np.linspace(0, np.pi, 500)
y_true = np.cos(3 * x_plot)

predictions = {}

for p in p_values:
    # Phi = generate_cos_vector(x_train, p)
    Phi = generate_weighted_cos_vector(x_train, p)
    b_hat = np.linalg.pinv(Phi) @ y_train
    # Phi_plot = generate_cos_vector(x_plot, p)
    Phi_plot = generate_weighted_cos_vector(x_plot, p)
    y_pred = Phi_plot @ b_hat
    predictions[p] = y_pred
    show_graph(p, x_train, y_train, x_plot, y_true, y_pred)