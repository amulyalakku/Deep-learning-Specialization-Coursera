# Improving Deep Neural Networks: Hyperparameter Tuning, Regularization and Optimization (Coursera)
## Andrew Ng's Deep Learning Specialization - Course 2

## Week 1: Practical Aspects of Deep Learning

This week focuses on the foundational setup and diagnostic tools for any deep learning project.

### 1.1 Setting Up Your Machine Learning Application

#### Train / Dev / Test Sets

The most fundamental step for an unbiased evaluation of your model's performance.

-   **Training Set:** The data the model learns from. All parameter optimization happens here.
-   **Development (Dev) Set:** Used to tune hyperparameters, select model architectures, and diagnose bias/variance. It acts as an unbiased proxy for the test set during development.
-   **Test Set:** Used **only once** at the very end to report the final, unbiased performance of the chosen model. Touching this set during development invalidates it.

**Split Ratios:**
-   **Small Data Era (<100k):** 60% Train / 20% Dev / 20% Test.
-   **Big Data Era (>1M):** 98% Train / 1% Dev / 1% Test. A 10,000-example dev/test set is often sufficient for statistical significance.

**Mismatched Data Distributions:**
-   **Rule:** Your **dev and test sets must come from the same distribution**—the one you expect in the real world. This ensures you are tuning and evaluating your model on the correct target. It's acceptable for the training data (e.g., web-scraped images) to differ from the dev/test data (e.g., user-uploaded mobile photos).

#### Bias and Variance: The Core Diagnostic Tool

This framework is your primary guide for what to do next in your iterative process.

-   **Bias (Underfitting):** The model is too simple for the data. It cannot even fit the training data well.
    -   **Symptom:** High Training Error.
-   **Variance (Overfitting):** The model is too complex. It has memorized the training data, including its noise, and fails to generalize to new data.
    -   **Symptom:** Low Training Error but High Dev Error.

**Diagnosing with Train/Dev Error:** (Assuming human/Bayes error is ~0%)

| Scenario                  | Train Set Error | Dev Set Error | Diagnosis                      | Primary Solution                                                                                                   |
| :------------------------ | :-------------- | :------------ | :----------------------------- | :----------------------------------------------------------------------------------------------------------------- |
| **High Bias**               | 15%             | 16%           | **Underfitting.** The model is fundamentally not powerful enough. | **Get a bigger model.** Add more layers or neurons, train longer, try a more advanced optimization algorithm (like Adam), or a different network architecture. |
| **High Variance**           | 1%              | 11%           | **Overfitting.** The model is too powerful for the amount of data it has. | **Add more data or regularize.** Get more training data, use data augmentation, or apply regularization techniques like L2, Dropout, or Early Stopping. |
| **High Bias & High Variance** | 15%             | 30%           | **Underfitting & Overfitting.** The model is a poor fit. The gap between Train and Dev error shows high variance. The high Train error itself shows high bias. | **First, fix the bias.** Use a bigger network. This will likely increase variance even more. **Then, fix the variance** using regularization. |
| **Low Bias & Low Variance** | 0.5%            | 1%            | **Good Fit.** This is the goal. | Deploy the model or try to push performance even further.                                                            |

### 1.2 Regularization: Combating Overfitting (High Variance)

#### L2 Regularization (Weight Decay)

-   **Intuition:** A complex model often has large parameter values. L2 regularization penalizes large weights, forcing the model to find a simpler representation.
-   **How it works:** Adds a penalty term to the cost function:
    `J_reg(W, b) = J(W, b) + (λ / 2m) * ||W||²_F`
    -   `λ` (lambda) is the regularization hyperparameter. A larger `λ` results in smaller weights (stronger regularization).
    -   `||W||²_F` is the Frobenius norm of the weight matrix (sum of squares of all elements).
-   **Effect on Backpropagation:** It adds `(λ/m) * W` to the gradient of the weights `dW`. This leads to the update rule `W := (1 - αλ/m) * W - α * (original dW)`, which effectively "decays" the weights at each step.

#### Dropout Regularization

-   **Intuition:** At each training iteration, randomly deactivate a fraction of neurons in a layer. This prevents co-adaptation, where neurons become too reliant on the specific outputs of other neurons. It forces the network to learn more robust, distributed features.
-   **How it works (Inverted Dropout):**
    1.  Define `keep_prob` (e.g., 0.8), the probability of keeping a neuron active.
    2.  Create a random binary mask `d[l]` of the same shape as the layer's activations `A[l]`.
    3.  Apply the mask: `A[l] = A[l] * d[l]`
    4.  **Scale the activations:** `A[l] /= keep_prob`. This crucial step ensures the expected output of the layer remains the same, so no changes are needed at test time.
-   **At Test Time:** **Never use dropout.** You use the full network to get the most accurate, deterministic prediction.

#### Other Regularization Methods
-   **Data Augmentation:** Creating new training examples from existing ones (e.g., flipping, rotating, cropping images).
-   **Early Stopping:** Monitor the dev set error during training. Stop training when the dev error starts to increase, as this indicates the model has begun to overfit.

### 1.3 Normalizing Inputs

-   **Why?** Normalizing your input features to have zero mean and unit variance can significantly speed up training.
-   **Intuition:** If your features are on very different scales (e.g., `age` from 20-70, `income` from 50k-200k), your cost function becomes a very elongated, narrow bowl. Gradient descent will oscillate back and forth and take a long, slow path to the minimum. Normalizing makes the cost function more symmetrical (like a round bowl), allowing gradient descent to take a much more direct path.
-   **How:**
    1.  Compute mean `μ` and variance `σ²` from the **training set**.
    2.  Subtract the mean and divide by the standard deviation: `X = (X - μ) / σ`.
    3.  **Important:** Use the *same* `μ` and `σ²` (calculated from the training set) to normalize your dev and test sets.

### 1.4 Vanishing / Exploding Gradients

-   **The Problem:** In very deep networks, the gradients can become extremely small (vanish) or extremely large (explode) as they are backpropagated through many layers. This makes training very difficult or impossible.
-   **Cause:** Repeated multiplication by weight matrices. If weights are consistently > 1, gradients explode. If weights are consistently < 1, gradients vanish.
-   **Solution: Careful Weight Initialization.**
    -   Don't initialize weights to zero (this breaks symmetry).
    -   Don't initialize them to be too large or too small.
    -   **He/Xavier Initialization:** A principled way to set the initial random weights. It sets the variance of the weights `W[l]` to be `sqrt(2 / n[l-1])` (for ReLU activations) or `sqrt(1 / n[l-1])` (for Tanh). This helps keep the signal flowing through the network at a stable scale.

### 1.5 Gradient Checking

-   **Purpose:** A crucial debugging technique to verify that your implementation of backpropagation is correct.
-   **Intuition:** The mathematical definition of a derivative is the slope of a function, which can be approximated numerically.
    -   `f'(θ) ≈ (f(θ + ε) - f(θ - ε)) / (2ε)`
-   **How it Works:**
    1.  "Unroll" all your parameters `W[1], b[1], W[2], b[2], ...` into a single giant vector `θ`.
    2.  For each element `θ_i` in `θ`:
        a.  Calculate the **numerical gradient `grad_approx[i]`** using the two-sided difference formula above. This is slow but simple.
        b.  Calculate the **analytical gradient `grad_backprop[i]`** from your backpropagation algorithm. This is fast but complex and prone to bugs.
    3.  **Compare the two gradients.** Calculate the Euclidean distance between the two vectors: `||grad_approx - grad_backprop|| / (||grad_approx|| + ||grad_backprop||)`.
        -   If the result is very small (e.g., < 10⁻⁷), your backprop is likely correct.
        -   If it's large (e.g., > 10⁻³), there is a bug in your implementation.
-   **Important Rules:**
    -   **Use it for debugging only, then turn it off.** It is far too slow to use during actual training.
    -   If it fails, isolate the error. Check the gradient for each parameter matrix (`dW[1]`, `db[1]`, etc.) individually.
    -   Remember to include the regularization term in your cost `J` when checking gradients.
    -   **Gradient checking does not work with dropout.** Turn off dropout before performing the check.

---

## Week 2: Optimization Algorithms

**Purpose:** To introduce algorithms that accelerate gradient descent, allowing for faster training.

### 2.1 Mini-Batch Gradient Descent

-   **Core Idea:** Instead of one update per epoch (Batch GD) or one update per example (Stochastic GD), find a middle ground.
-   **Process:**
    1.  Shuffle the training set.
    2.  Partition it into mini-batches of a chosen size (e.g., 64, 128, 512, 1024).
    3.  Iterate through the mini-batches, performing one step of gradient descent on each.
-   **Why it's the standard:** It provides a balance between the smooth convergence of Batch GD and the speed of SGD, while fully leveraging modern hardware's ability to vectorize computations.

### 2.2 Exponentially Weighted Averages (EWA)

-   **The concept underlying Momentum and RMSprop.** It's a way to compute a moving average.
-   **Formula:** `v_t = β * v_{t-1} + (1 - β) * θ_t`
    -   `θ_t` is the current value (e.g., today's temperature, the current gradient).
    -   `β` controls the size of the averaging window. `β=0.9` averages over ~10 data points. `β=0.98` averages over ~50 data points.

### 2.3 Gradient Descent with Momentum

-   **How it works:** Applies an EWA to the gradients. It uses the *velocity* of the gradients, not just the current gradient, for the update.
-   **Effect:** It dampens oscillations in directions of high curvature and accelerates movement in the consistent downhill direction.

### 2.4 RMSprop (Root Mean Square Prop)

-   **How it works:** Applies an EWA to the *squared* gradients. It adapts the learning rate per-parameter.
-   **Effect:** It divides the update by a measure of the recent gradient magnitude. This means parameters with large gradients get smaller updates (to prevent overshooting), and parameters with small gradients get larger updates (to speed up learning).

### 2.5 Adam Optimization Algorithm

-   **What it is:** The "Adaptive Moment Estimation" algorithm. It's the most widely used and recommended optimizer.
-   **How it works:** It combines the best of both worlds:
    1.  It uses an EWA on the gradient itself (the **first moment**, like Momentum).
    2.  It uses an EWA on the squared gradient (the **second moment**, like RMSprop).
    3.  It uses both to perform an adaptive per-parameter update.
-   It includes a bias-correction step to handle the initial iterations of the EWA, making it more stable early in training.

---

## Week 3: Hyperparameter Tuning, Batch Normalization, and Softmax

### 3.1 Hyperparameter Tuning

-   **Process:** An iterative, empirical science.
-   **Random Search > Grid Search:** Don't use grid search. Randomly sampling hyperparameters is more efficient because it explores more unique values for the most important parameters.
-   **Coarse-to-Fine Search:** Start with a broad random search. Once you find a promising region of the hyperparameter space, perform a finer-grained random search in that smaller region.
-   **Sampling on the Right Scale:**
    -   **Learning Rate (`α`):** Sample on a **log scale**. E.g., `10^-4` to `10^-1`.
    -   **Momentum Beta (`β`):** Sample `1-β` on a log scale (e.g., from `10^-3` to `10^-1`), then calculate `β = 1 - (sampled_value)`. This effectively samples values like 0.9, 0.99, 0.999 more evenly.

### 3.2 Batch Normalization (Batch-Norm)

-   **The Problem it Solves (Internal Covariate Shift):** The distribution of a hidden layer's inputs changes during training because the parameters of previous layers are changing. This makes the learning task for that layer much harder.
-   **The Solution:** Normalize the pre-activation values `Z[l]` within each mini-batch to have zero mean and unit variance.
    1.  Calculate mini-batch mean `μ` and variance `σ²`.
    2.  Normalize: `Z_norm = (Z - μ) / sqrt(σ² + ε)`.
    3.  **Crucially, re-scale and shift:** `Z_tilde = γ * Z_norm + β`. Here, `γ` and `β` are **learnable parameters** of the model, just like `W` and `b`. This allows the network to learn the optimal mean and variance for each layer's activations. It doesn't have to be zero mean and unit variance if that's not optimal.
-   **Why it's so effective:**
    1.  **Speeds up training dramatically.**
    2.  Makes the network much less sensitive to weight initialization.
    3.  Provides a slight regularization effect.

### 3.3 Multi-class Classification and Softmax

-   **When to use:** When your output `y` can be one of C > 2 classes.
-   **Architecture:** The output layer `L` has `C` neurons.
-   **Softmax Activation Function:**
    `a_i[L] = e^(z_i[L]) / Σ_{j=1 to C} e^(z_j[L])`
    -   It takes the raw scores `Z[L]` and transforms them into a probability distribution that sums to 1. `a_i[L]` can be interpreted as `P(y=i | X)`.
-   **Loss Function:** Use **Categorical Cross-Entropy Loss**.

### 3.4 Deep Learning Frameworks

-   The course ends by acknowledging that while building these components from scratch is essential for understanding, in practice, you'll use a framework like TensorFlow, PyTorch, or Keras.
-   These frameworks have pre-built, highly-optimized implementations of all the concepts covered: layers, activation functions, optimizers, regularization, and more. Your job becomes defining the architecture and then letting the framework handle the complex underlying math and execution.
