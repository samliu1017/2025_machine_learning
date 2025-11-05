# Assignment 1
---
### 1.Evaluate $\theta^1$
根據題目中的模型： $h(x_1,x_2) = \sigma(b+w_1x_1+w_2x_2)$, Sigmoid function: $\sigma(z)=\frac{1}{1+e^{-z}}$
Consider MSE loss: $L=\tfrac12(y-h)^2 = \tfrac12(h-y)^2$
And we find the gradient:
$$
\frac{\partial L}{\partial b}=(h-y)\sigma'(z), \quad
\frac{\partial L}{\partial w_i}=(h-y)\sigma'(z)\,x_i
$$
where
$$
\sigma'(z) = \sigma(z)(1-\sigma(z)), \quad z=b+w_1x_1+w_2x_2
$$
根據題目，帶入數值 $(x_1,x_2,y)=(1,2,3)$， $\theta^0 = (b,w_1,w_2)=(4,5,6)$。
算出 $z=21$
使用公式$\theta ^1 = \theta ^0 - \alpha \partial_\theta Loss$
可以求出 $\theta^1 = (b^1, x_1^1, x_2^2)$ ：
$$
\begin{aligned}
b^{1}   &= 4-\alpha\bigl(\sigma(21)-3\bigr)\,\sigma(21)\bigl(1-\sigma(21)\bigr)\cdot 1,\\[6pt]
w_1^{1} &= 5-\alpha\bigl(\sigma(21)-3\bigr)\,\sigma(21)\bigl(1-\sigma(21)\bigr)\cdot 1,\\[6pt]
w_2^{1} &= 6-\alpha\bigl(\sigma(21)-3\bigr)\,\sigma(21)\bigl(1-\sigma(21)\bigr)\cdot 2.
\end{aligned}
$$
---

### 2.(a) Find the expression
Let $\sigma(x) = \frac{1}{1+e^{-x}}$
Then：
$\sigma'(x) = \frac{e^{-x}}{(1+e^{-x})^2} = \sigma(1-\sigma)$
$\sigma''(x) = \frac{d}{dx} [\sigma(1-\sigma)] = \sigma'[1-2\sigma] = \sigma(1-\sigma)(1-2\sigma)$
$\sigma^{(3)}(x) = \frac{d}{dx}[\sigma''] = \frac{d}{dx}[2\sigma^3 -3\sigma^2 + \sigma] = \sigma'(6\sigma^2 -6 \sigma + 1) = \sigma(1-\sigma)(6\sigma^2 -6 \sigma + 1)$

---

### 2.(b) Find the relation between sigmoid function and hyperbolic function.
The sigmoid function is $\sigma(x) = \frac{1}{1+e^{-x}}$
The hyperbolic sin function is $\sinh(x) = \frac{e^x - e^{-x}}{2}$.
The hyperbolic cos function is $\cosh(x) = \frac{e^x + e^{-x}}{2}$ .
The hyperbolic tan function is $\tanh(x) = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}$.
Because $\sigma(x) = \frac{1}{1+e^{-x}} = \frac{e^{x}}{1+e^{x}} = 1 - \frac{1}{1+e^{x}}$
So, $ e^{x} = \frac{1}{1-\sigma} -1 = \frac{\sigma}{1-\sigma}$
#### (1)sinh
$$\sinh(x) = \frac{e^x - e^{-x}}{2} = \frac{\frac{\sigma}{1-\sigma} - \frac{1-\sigma}{\sigma}}{2} = \frac{2\sigma-1}{2\sigma(1-\sigma)}$$
#### (2)cosh
$$\cosh(x) = \frac{e^x + e^{-x}}{2} = \frac{\frac{\sigma}{1-\sigma} + \frac{1-\sigma}{\sigma}}{2} = \frac{2\sigma^2 - 2\sigma + 1}{2\sigma(1-\sigma)}$$
#### (3)tanh
$$
\tanh\left(x\right) = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}} = \frac{e^{2x} - 1}{e^{2x} + 1}
$$
$$
\sigma(2x) = \frac{1}{1+e^{-2x}} = \frac{e^{2x}}{e^{2x} + 1}, \qquad 2\sigma(2x) - 1 = \frac{2e^{2x}}{e^{2x} + 1}-\frac{e^{2x}+1}{e^{2x} + 1} = \frac{e^{2x} - 1}{e^{2x} + 1}
$$
So,
$$
\sigma(x) = \frac{1}{2}\left(1 + \tanh\left(\frac{x}{2}\right)\right), \qquad \tanh\left(x\right) = 2\sigma(2x) - 1
$$

---

### 3. Some Questions
#### Q1. Why do we usually begin with a linear model?
##### Answer 1 (By Gemini)

Key Advantages of Linear Models
Linear models are valuable for several reasons, which make them a popular choice for a wide range of applications:
$(1)$ Simplicity and Interpretability
$(2)$ Computational Efficiency
$(3)$ Well-Understood Statistical Properties
$(4)$ Foundation for More Complex Models

When to Use a Linear Model
A linear model is often the best choice in the following situations:
$(1)$ When the relationship is likely linear
$(2)$ For baseline performance
$(3)$ When interpretability is a priority
Limitations of Linear Models

Their main limitation is their assumption of a linear relationship between the dependent and independent variables.
If the underlying data has a complex, non-linear pattern, a linear model will likely fail to capture it accurately. 
This can lead to underfitting and poor predictive performance.

#### Q2. What is the difference between BGD, SGD, and Mini-Batch Gradient Descent, and what are their advantages?

##### Answer 2 (By Chat GPT)
$(1)$ Batch Gradient Descent (BGD)

Advantages:
Stable convergence because the gradient is computed using all data.
Smooth and predictable updates.

Disadvantages:
Very slow on large datasets since every step requires going through all data.
Not suitable for online or streaming data.
Use Case: Small datasets

$(2)$ Stochastic Gradient Descent (SGD)

Advantages:
Much faster per update because it uses only one example.
Can handle very large datasets or streaming data.
Can escape local minima more easily due to noisy updates.

Disadvantages:
Updates are noisy and can fluctuate heavily.
Convergence is less stable; may require learning rate decay.
Use Case: Very large/streaming data

$(3)$ Mini-Batch Gradient Descent

Advantages:
Faster than BGD because it doesn’t need the full dataset.
Less noisy than SGD; more stable convergence.
Can exploit optimized matrix operations on hardware (like GPUs).
Strikes a balance between speed and accuracy.

Disadvantages:
Still requires tuning of batch size.
Can have some noise in gradient estimate, though usually beneficial.
Use Case: Most deep learning tasks