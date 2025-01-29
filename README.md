# European Option Pricing with COS Method

## Overview
This project implements the **COS Method** for pricing European call and put options efficiently. The **COS Method**, introduced by Fang and Oosterlee (2008), is a spectral expansion approach that significantly improves computational efficiency compared to traditional numerical integration methods.

## Author
**Shubh Shrishrimal**  
Email: shubhshrishrimal1125@gmail.com  

## Table of Contents
- [Introduction to European Options](#introduction-to-european-options)
- [Mathematical Foundation](#mathematical-foundation)
- [COS Method Explained](#cos-method-explained)
- [Project Structure](#project-structure)
- [Usage Guide](#usage-guide)
- [Results and Performance](#results-and-performance)
- [License](#license)

## Introduction to European Options
A **European option** is a financial derivative that gives the holder the right, but not the obligation, to buy (Call) or sell (Put) an underlying asset at a predetermined strike price on the option's expiration date.

The Black-Scholes formula is a popular method for pricing European options, but it requires assumptions that may not always hold in real markets. The **COS method** is an alternative that efficiently computes option prices for a broader range of stochastic processes.

## Mathematical Foundation
The price of a European option is given by the risk-neutral expectation:
\[
    V(S_0, t) = e^{-r(T-t)} \mathbb{E} [ (S_T - K)^+ ]
\]
where:
- \( S_0 \) is the current asset price,
- \( K \) is the strike price,
- \( T \) is the time to maturity,
- \( r \) is the risk-free interest rate,
- \( S_T \) is the asset price at maturity,
- \( (x)^+ \) represents the max(x,0) function.

Instead of solving this integral directly, the **COS method** uses a Fourier cosine expansion to approximate it efficiently.

## COS Method Explained
The COS method is based on expanding the option payoff function into a Fourier cosine series:
\[
    f(x) \approx \sum_{n=0}^{N-1} A_n \cos \left( \frac{n \pi (x - a)}{b - a} \right)
\]
where:
- \( a, b \) define the truncation range,
- \( A_n \) are the cosine series coefficients derived from the characteristic function \( \varphi(u) \),
- \( N \) is the number of terms in the expansion.

The option price is then computed using:
\[
    V(S_0, t) = e^{-r(T-t)} \sum_{n=0}^{N-1} A_n \text{Re} \left[ \varphi \left( \frac{n \pi}{b - a} \right) \right]
\]
This method is fast because it exploits the efficiency of Fourier series and the availability of closed-form expressions for many characteristic functions.

## Project Structure
```
European-Option-Pricing-with-COS-Method/
│── Pricing of European Call and Put options with the COS method.ipynb  # Main notebook
│── app.py  # Streamlit web app for option pricing
│── helper.py  # Helper functions for COS method computation
│── README.md  # Project documentation
```

### Key Components
1. **Pricing of European Call and Put options with the COS method.ipynb**
   - Implements the COS method step by step.
   - Includes theoretical background and Python code.
   - Compares COS pricing with Black-Scholes results.

2. **app.py**
   - A **Streamlit web application** for interactive option pricing.
   - Users can input stock price, strike price, volatility, risk-free rate, and maturity.
   - Computes and displays option prices.

3. **helper.py**
   - Contains utility functions for computing the COS method expansion.
   - Includes characteristic function implementations.

## Usage Guide
### Running the Jupyter Notebook
To execute the COS method for European option pricing, run:
```bash
jupyter notebook "Pricing of European Call and Put options with the COS method.ipynb"
```

### Running the Streamlit App
To launch the interactive web application, run:
```bash
streamlit run app.py
```

### Example Usage
Import the COS method and compute an option price:
```python
from helper import cos_method

S0 = 100    # Initial stock price
K = 100     # Strike price
T = 1       # Time to maturity
r = 0.05    # Risk-free rate
sigma = 0.2 # Volatility
N = 64      # Number of expansion terms

price = cos_method(S0, K, T, r, sigma, N)
print(f"Option Price: {price}")
```

## Results and Performance
The COS method provides highly accurate results with significantly fewer function evaluations compared to Monte Carlo or finite difference methods. It is especially efficient for pricing options under models with known characteristic functions, such as:
- Black-Scholes model
- Heston model (stochastic volatility)
- Variance Gamma model

## License
This project is licensed under the **MIT License**. See the `LICENSE` file for details.

---
**References:**  
Fang, F., & Oosterlee, C. W. (2008). A novel pricing method for European options based on Fourier-cosine series expansions. SIAM Journal on Scientific Computing, 31(2), 826-848.

