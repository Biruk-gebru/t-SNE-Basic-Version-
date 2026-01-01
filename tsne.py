"""
t-SNE Implementation (Basic Version)
------------------------------------
This script implements the t-Distributed Stochastic Neighbor Embedding (t-SNE) algorithm
from scratch using only Python and NumPy, as per the assignment requirements.

Overview of Functions:
1. calculate_high_dim_prob(X, perplexity):
   - Input: High-dimensional data matrix X (N x D), perplexity parameter.
   - Output: Symmetrized probability matrix P (N x N).
   - Details: Converts Euclidean distances to conditional probabilities using Gaussian kernels.
     Matches the entropy of the distribution to the log(perplexity) via binary search for sigma.
     Returns the symmetrized joint probability matrix P_ij = (P_{j|i} + P_{i|j}) / 2N.

2. calculate_low_dim_prob(Y):
   - Input: Low-dimensional map Y (N x 2).
   - Output: Q matrix (N x N) and the unnormalized numerator array.
   - Details: Uses a Student t-distribution (1 degree of freedom) to compute probabilities
     in the embedding space. This heavy-tailed distribution helps alleviate the crowding problem.

3. tsne(X, n_components, perplexity, n_iter, learning_rate):
   - Input: Data X, target dimensions, perplexity, iterations, learning rate.
   - Output: Final embedding Y.
   - Details: Performs gradient descent on the Kullback-Leibler (KL) divergence cost function.
     Includes momentum (0.5 initially, then 0.8) and adaptive learning rate gains to speed up convergence.
     Applies "early exaggeration" (P * 4) for the first 100 iterations to encourage cluster separation.

Implementation Choices & Limitations:
- Initialization: Random small Gaussian noise.
- Optimization: Standard Gradient Descent with momentum.
- Limitations: O(N^2) complexity means this is slow for N > 2000.
- Analysis: Works well on MNIST subset but requires tuning perplexity (default 30).
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits 

def calculate_high_dim_prob(X, perplexity=30.0):
    n = X.shape[0]
    # Compute pairwise squared Euclidean distances
    sum_X = np.sum(np.square(X), axis=1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    
    P = np.zeros((n, n))
    beta = np.ones((n, 1)) # beta = 1 / (2 * sigma^2)
    logU = np.log(perplexity)
    
    # Binary search for each point's optimal sigma
    for i in range(n):
        beta_min = 0.0
        beta_max = np.inf
        
        # Exclude self-distance
        Di = np.delete(D[i, :], i)
        
        for _ in range(50):
            Hbeta = np.exp(-Di * beta[i])
            sum_Hbeta = np.sum(Hbeta) + 1e-12
            
            # Entropy calculation
            H = np.log(sum_Hbeta) + beta[i] * np.sum(Di * Hbeta) / sum_Hbeta
            diff = H - logU
            
            if np.abs(diff) < 1e-5:
                break
            
            if diff > 0:
                beta_min = beta[i]
                if beta_max == np.inf:
                    beta[i] *= 2
                else:
                    beta[i] = (beta[i] + beta_max) / 2
            else:
                beta_max = beta[i]
                beta[i] = (beta[i] + beta_min) / 2
                
        # Compute final row probabilities
        Hbeta = np.exp(-D[i, :] * beta[i])
        Hbeta[i] = 0
        P[i, :] = Hbeta / (np.sum(Hbeta) + 1e-12)
        
    # Symmetrize and normalize
    P = (P + P.T) / (2 * n)
    return np.maximum(P, 1e-12)

def calculate_low_dim_prob(Y):
    # Student t-distribution (1 / (1 + dist^2))
    sum_Y = np.sum(np.square(Y), axis=1)
    num = 1 / (1 + np.add(np.add(-2 * np.dot(Y, Y.T), sum_Y).T, sum_Y))
    np.fill_diagonal(num, 0)
    Q = num / np.sum(num)
    return np.maximum(Q, 1e-12), num

def tsne(X, n_components=2, perplexity=30.0, n_iter=1000, learning_rate=200.0):
    n, d = X.shape
    
    print("Computing pairwise affinities...")
    P = calculate_high_dim_prob(X, perplexity)
    
    # Early exaggeration
    P = P * 4
    
    # Initialize Y
    Y = np.random.randn(n, n_components) * 1e-4
    
    dY = np.zeros_like(Y)
    iY = np.zeros_like(Y)
    gains = np.ones_like(Y)
    
    print(f"Starting gradient descent ({n_iter} iterations)...")
    for i in range(n_iter):
        Q, num = calculate_low_dim_prob(Y)
        
        # Calculate gradient: 4 * sum((P - Q) * num * (Yi - Yj))
        PQ = P - Q
        # Vectorized gradient calculation
        # The equation for gradient is: dC/dy_i = 4 * sum_j (p_ij - q_ij)(1 + ||y_i - y_j||^2)^-1 (y_i - y_j)
        # We can use NumPy broadcasting -> (P - Q) * num is a matrix scaling factor for pairwise vectors
        for j in range(n):
            dY[j, :] = 4 * np.sum((PQ[j, :] * num[j, :])[:, np.newaxis] * (Y[j, :] - Y), axis=0)
            
        # Momentum adjustment
        if i < 250:
            momentum = 0.5
        else:
            momentum = 0.8
            
        # Stop early exaggeration
        if i == 100:
            P = P / 4
            
        # Adaptive learning rate (gains)
        gains = (gains + 0.2) * ((dY > 0) != (iY > 0)) + \
                (gains * 0.8) * ((dY > 0) == (iY > 0))
        gains[gains < 0.01] = 0.01
        
        # Update using momentum and gradient
        iY = momentum * iY - learning_rate * (gains * dY)
        Y = Y + iY
        Y = Y - np.mean(Y, axis=0) # Re-center
        
        if (i + 1) % 100 == 0:
            error = np.sum(P * np.log(P / Q))
            print(f"Iteration {i+1}, Error: {error:.4f}")
            
    return Y

if __name__ == "__main__":
    # Load data
    digits = load_digits()
    X = digits.data
    y = digits.target
    
    # Use subset for demonstration speed
    n_samples = 500
    print(f"Running t-SNE on {n_samples} MNIST digits...")
    
    X_subset = X[:n_samples] / 16.0 # Normalize 0-1
    y_subset = y[:n_samples]
    
    Y = tsne(X_subset)
    
    # Visualization
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(Y[:, 0], Y[:, 1], c=y_subset, cmap='tab10', s=15)
    plt.legend(*scatter.legend_elements(), title="Digit Class")
    plt.title("t-SNE Visualization (NumPy Implementation)")
    plt.savefig("tsne_result.png")
    print("Plot saved to tsne_result.png")
