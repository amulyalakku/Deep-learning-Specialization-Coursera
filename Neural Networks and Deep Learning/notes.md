# Neural Networks and Deep Learning (Coursera)
## Andrew Ng's Deep Learning Specialization - Course 1

### Course Philosophy: Building from the Ground Up

This course demystifies deep learning by building a neural network from scratch. The core philosophy is to understand every mathematical component—from a single neuron (logistic regression) to a full, multi-layer network. By the end, you will not just be a user of a deep learning library; you will understand what is happening under the hood, which is critical for effective model building and debugging.

---

## Week 1: Introduction to Deep Learning

**Purpose:** To set the stage, introduce the concept of a neural network, and show how it powers modern AI.

### What is a Neural Network?

-   **Core Idea:** A neural network is a powerful function approximator, inspired by the structure of the human brain. It learns to map inputs (`X`) to outputs (`y`) by adjusting its internal parameters.
-   **Structure:** It's composed of layers of "neurons." Each neuron is a simple computational unit.
-   **Example: Housing Price Prediction**
    -   **Inputs (Features `X`):** Size of the house, number of bedrooms, zip code, wealth of the neighborhood.
    -   **Hidden Layers:** The network might learn intermediate "concepts" on its own.
        -   The first hidden layer might learn simple features like "family size" (from bedrooms and size) or "school quality" (from zip code and wealth).
        -   A later hidden layer might combine these to learn a more abstract concept like "family-friendliness."
    -   **Output (`ŷ`):** The predicted price.
-   **Key Takeaway:** The power of deep learning comes from the network's ability to **automatically learn a hierarchy of features**, from simple to complex, without needing explicit programming.

---

## Week 2: Neural Networks Basics

**Purpose:** To build the fundamental building block of a neural network: a single neuron, which is mathematically equivalent to **Logistic Regression**.

### Logistic Regression as a Neural Network

-   **Goal:** Binary classification (output is 0 or 1). For example, given an image, is it a cat?
-   **The "Neuron" Model:**
    1.  **Input:** A feature vector `x`. For an image, this is a "flattened" vector of all its pixel values. If the image is 64x64x3 (height x width x color channels), `x` will be a vector of size 12,288.
    2.  **Parameters:** The neuron has its own weights `w` (a vector of the same size as `x`) and a bias `b` (a scalar).
    3.  **Linear Step:** Compute `z = w^T * x + b`. This is just a linear function.
    4.  **Activation Step:** Pass `z` through an **activation function**. For logistic regression, this is the **sigmoid function**, `σ(z) = 1 / (1 + e^(-z))`.
        -   **Purpose of Sigmoid:** It squashes any real-valued number `z` into a range between 0 and 1. This is perfect for representing a probability.
    5.  **Output (`ŷ`):** `ŷ = a = σ(z)`. This is the predicted probability that the output is 1 (e.g., `P(y=1|x)`).

### The Cost Function

-   **Goal:** We need a way to measure how "wrong" our model's prediction `ŷ` is compared to the true label `y`.
-   **Why Not Squared Error?** Using squared error `(ŷ - y)²` creates a non-convex optimization problem with many local minima for logistic regression, making it hard for gradient descent to find the global minimum.
-   **The Solution: Log Loss (or Binary Cross-Entropy)**
    -   **Loss Function (for a single example):** `L(ŷ, y) = - [ y * log(ŷ) + (1-y) * log(1-ŷ) ]`
    -   **Intuition:**
        -   If the true label `y = 1`, the loss is `-log(ŷ)`. We want `ŷ` to be close to 1, which makes `-log(ŷ)` close to 0 (good). If `ŷ` is close to 0, `-log(ŷ)` becomes very large (bad).
        -   If the true label `y = 0`, the loss is `-log(1-ŷ)`. We want `ŷ` to be close to 0, which makes `-log(1-ŷ)` close to 0 (good). If `ŷ` is close to 1, `-log(1-ŷ)` becomes very large (bad).
-   **Cost Function (for the entire dataset):** The cost `J` is just the average of the loss over all `m` training examples.
    `J(w, b) = (1/m) * Σ L(ŷ_i, y_i)`

### Gradient Descent: The Learning Algorithm

-   **Goal:** To find the optimal values of `w` and `b` that minimize the cost function `J`.
-   **Analogy:** Imagine you are on a foggy mountain and want to get to the lowest point. You can't see the whole landscape, but you can feel the slope right where you are. You take a step in the steepest downhill direction. You repeat this process until you reach a valley.
-   **The Process:**
    1.  Initialize `w` and `b` randomly.
    2.  Repeat for a number of iterations:
        a.  Compute the gradients (derivatives) of the cost function with respect to the parameters: `dw = dJ/dw` and `db = dJ/db`. These gradients tell you the "slope" of the cost function.
        b.  **Update the parameters:** Move them in the opposite direction of the gradient.
            -   `w := w - α * dw`
            -   `b := b - α * db`
        c.  `α` (alpha) is the **learning rate**, a hyperparameter that controls how big of a step you take.

### Derivatives and the Computation Graph

-   **Key Idea:** The **chain rule** from calculus is the engine of backpropagation. A neural network is just a series of nested functions. To find the derivative of the final cost with respect to an early parameter, you multiply the derivatives of each function along the path from the output back to that parameter.
-   **Computation Graph:** A visual way to represent the forward flow of computation and the backward flow of derivatives.
    -   **Example:** For `J(a,b,c) = 3 * (a + bc)`
        -   **Forward Pass:** Calculate `u = bc`, then `v = a + u`, then `J = 3v`.
        -   **Backward Pass (Backpropagation):**
            1.  `dJ/dv = 3`
            2.  `dJ/du = dJ/dv * dv/du = 3 * 1 = 3`
            3.  `dJ/da = dJ/dv * dv/da = 3 * 1 = 3`
            4.  `dJ/db = dJ/du * du/db = 3 * c`
            5.  `dJ/dc = dJ/du * du/dc = 3 * b`

---

## Week 3: Shallow Neural Networks

**Purpose:** To extend the single neuron model to a full, two-layer neural network.

### Two-Layer Neural Network Architecture

-   **Input Layer:** The feature vector `X`.
-   **Hidden Layer:** The first layer of neurons that performs computations. It is "hidden" because its true values are not observed in the training data.
-   **Output Layer:** The final neuron(s) that produce the prediction `ŷ`.
-   **Notation:**
    -   `[l]` denotes the layer number.
    -   `W[1]`, `b[1]` are the parameters for the hidden layer.
    -   `W[2]`, `b[2]` are the parameters for the output layer.

### Forward Propagation in a 2-Layer Network

For a single example `x`:
1.  **Hidden Layer Calculation:**
    -   `Z[1] = W[1] * x + b[1]`
    -   `A[1] = g[1](Z[1])` where `g[1]` is the activation function for the hidden layer.
2.  **Output Layer Calculation:**
    -   `Z[2] = W[2] * A[1] + b[2]`
    -   `A[2] = g[2](Z[2]) = σ(Z[2]) = ŷ` where `g[2]` is the activation for the output layer (sigmoid for binary classification).

### Activation Functions

-   **Why we need them:** If we only used linear operations, a deep network would just be a complex linear function. It wouldn't be any more powerful than a single layer. Activation functions introduce **non-linearity**, allowing the network to learn much more complex patterns.
-   **Common Choices for Hidden Layers:**
    -   **Sigmoid:** `σ(z) = 1 / (1 + e^(-z))`. **Problem:** Gradients are close to zero for large positive or negative inputs ("vanishing gradients"), which can stall learning. **Avoid using in hidden layers.**
    -   **Tanh (Hyperbolic Tangent):** `tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))`. Squashes values to a range of [-1, 1]. It's zero-centered, which often helps learning. Suffers from the same vanishing gradient problem as sigmoid.
    -   **ReLU (Rectified Linear Unit):** `ReLU(z) = max(0, z)`.
        -   **Why it's great:** It's simple, computationally fast, and doesn't have the vanishing gradient problem for positive inputs. **This is the most common and recommended default choice.**
    -   **Leaky ReLU:** `LeakyReLU(z) = max(0.01*z, z)`. A small modification to ReLU to fix the "dying ReLU" problem where neurons can get stuck in the zero-gradient region for negative inputs.

---

## Week 4: Deep Neural Networks

**Purpose:** To generalize the two-layer network to an L-layer deep neural network.

### What is a "Deep" Neural Network?

A neural network with many hidden layers.

-   **Why Deep? The Hierarchy of Features:**
    -   Early layers tend to learn simple features (edges, color gradients).
    -   Middle layers compose these simple features into more complex ones (eyes, noses, textures).
    -   Late layers compose those into even more complex concepts (faces, objects).
-   This compositional structure allows deep networks to learn incredibly complex functions efficiently. A very deep but "thin" (few neurons per layer) network can often learn functions that a very wide but shallow network cannot.

### Forward and Backward Propagation in an L-Layer Network

This is a generalization of the 2-layer case.

#### Forward Propagation

```python
# For loop from l = 1 to L
Z[l] = W[l] * A[l-1] + b[l]
A[l] = g[l](Z[l])
(Where A[0] = X)
````
#### Backpropagation: The Four Key Equations
This is the core of training. It calculates the gradients for each parameter.

Gradient of the final layer's pre-activation:
dZ[L] = A[L] - Y
Gradient of a layer's weights:
dW[l] = (1/m) * dZ[l] * A[l-1].T
Gradient of a layer's bias:
db[l] = (1/m) * np.sum(dZ[l], axis=1, keepdims=True)
Gradient of a hidden layer's pre-activation (the chain rule step):
dZ[l] = W[l+1].T * dZ[l+1] * g'[l](Z[l])
g'[l](Z[l]) is the derivative of the activation function of layer l.

The Big Picture: The process starts at the end (computing dZ[L]). Then it iterates backward (l = L-1, L-2, ..., 1), using the gradient from the next layer (dZ[l+1]) to compute the gradient for the current layer (dZ[l]). At each step, it also computes the gradients for the parameters dW[l] and db[l].

#### Parameters vs. Hyperparameters
This is a crucial distinction for practical ML.

**Parameters**: These are the values the model learns on its own through gradient descent.
Examples: The weights W and biases b.
**Hyperparameters**: These are the settings that you, the developer, choose to control the learning process. They are not learned by the model.
Examples:
Learning rate α
Number of iterations
Number of hidden layers L
Number of neurons per layer
Choice of activation function (ReLU, Tanh, etc.)
Regularization parameter λ
Dropout keep_prob
Mini-batch size
Adam optimizer parameters (β₁, β₂)

The workflow of applied deep learning is largely about choosing the right hyperparameters to help your model learn the best parameters. 

