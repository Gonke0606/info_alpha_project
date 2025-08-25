import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

# ハイパーパラメータ
NUM_TRAIN = 10      # 学習データ数
NUM_TEST = 50      # テストデータ数
DIM_X = 10        # d次元の確率ベクトルx
DIM_P = range(1, 20) # p次元のパラメータβ
NOISE_LEVEL = 10 # ノイズの大きさ

activation_functions = ["ReLU", "Sigmoid", "tanh", "GeLU", "Swish", "ReakyReLU"]
results = {func: [] for func in activation_functions}

def ReLU(x):
    return np.maximum(0, x)

def Sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def GeLU(x):
    return 0.5 * x * (1 + erf(x / np.sqrt(2)))

def Swish(x):
    return x * Sigmoid(x)

def ReakyReLU(x):
    return np.maximum(0.01 * x, x)

def generate_prob_vector(dim):
    x = np.random.rand(dim)
    return x / np.sum(x)

def generate_mean_zero_vector(num_data):
    x = np.random.randn(num_data)
    return x - np.mean(x)

def show_graph(errors, func_type):
    plt.figure(figsize=(10, 7))
    plt.plot(DIM_P, errors, label=func_type, lw=2, color='blue')
    plt.axvline(x=NUM_TRAIN, color='red', linestyle='--', label=f'p=n={NUM_TRAIN}')
    plt.title(f"", fontsize=16)
    plt.xlabel("num_parameters", fontsize=12)
    plt.ylabel("MSE", fontsize=12)
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.yscale("log")
    plt.show()

for func_type in activation_functions:
    test_errors = []

    for dim_p in DIM_P:
        W = np.random.randn(dim_p, DIM_X) / np.sqrt(DIM_X) # Wを正規化することによって、p=nの値の発散を抑制する
        b = np.random.randn(dim_p)

        X_train = np.array([generate_prob_vector(DIM_X) for i in range(NUM_TRAIN)])

        if func_type == "ReLU": Phi_train = ReLU(X_train @ W.T)
        elif func_type == "Sigmoid": Phi_train = Sigmoid(X_train @ W.T)
        elif func_type == "tanh": Phi_train = tanh(X_train @ W.T)
        elif func_type == "GeLU": Phi_train = GeLU(X_train @ W.T)
        elif func_type == "Swish": Phi_train = Swish(X_train @ W.T)
        elif func_type == "ReakyReLU": Phi_train = ReakyReLU(X_train @ W.T)

        noise_train = generate_mean_zero_vector(NUM_TRAIN) * NOISE_LEVEL
        y_train = Phi_train @ b + noise_train

        X_test = np.array([generate_prob_vector(DIM_X) for _ in range(NUM_TEST)])

        if func_type == "ReLU": Phi_test = ReLU(X_test @ W.T)
        elif func_type == "Sigmoid": Phi_test = Sigmoid(X_test @ W.T)
        elif func_type == "tanh": Phi_test = tanh(X_test @ W.T)
        elif func_type == "GeLU": Phi_test = GeLU(X_test @ W.T)
        elif func_type == "Swish": Phi_test = Swish(X_test @ W.T)
        elif func_type == "ReakyReLU": Phi_test = ReakyReLU(X_test @ W.T)

        noise_test = generate_mean_zero_vector(NUM_TEST) * NOISE_LEVEL
        y_test_true = Phi_test @ b + noise_test

        b_hat = np.linalg.pinv(Phi_train) @ y_train
        y_pred = Phi_test @ b_hat
        error = np.mean((y_test_true - y_pred)**2)
        test_errors.append(error)

    results[func_type] = test_errors

for func_type, errors in results.items():
    show_graph(errors, func_type)