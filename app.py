import streamlit as st
import numpy as np
import plotly.graph_objects as go
import time
import scipy.stats as stats
import pandas as pd

# Helper functions (keep these as they are in your current app.py)
def CallPutCoefficients(Cp, a, b, k):
    if str(Cp).lower()=='c' or str(Cp).lower()=="1":
        c=0.0
        d=b
        coef=Chi_Psi(a,b,c,d,k)
        Chi_k=coef['chi']
        Psi_k=coef['psi']
        if a<b and b <0.0:
            H_k=np.zeros([len(k),1])
        else:
            H_k=2.0/(b-a)*(Chi_k-Psi_k)
    elif str(Cp).lower()=="p" or str(Cp).lower()=="-1":
        c = a
        d = 0.0
        coef = Chi_Psi(a,b,c,d,k)
        Chi_k = coef["chi"]
        Psi_k = coef["psi"]
        H_k      = 2.0 / (b - a) * (- Chi_k + Psi_k)               
    return H_k

def Chi_Psi(a, b, c, d, k):
    psi = np.sin(k * np.pi * (d - a) / (b - a)) - np.sin(k * np.pi * (c - a)/(b - a))
    psi[1:] = psi[1:] * (b - a) / (k[1:] * np.pi)
    psi[0] = d - c
    
    chi = 1.0 / (1.0 + np.power((k * np.pi / (b - a)) , 2.0)) 
    expr1 = np.cos(k * np.pi * (d - a)/(b - a)) * np.exp(d)  - np.cos(k * np.pi 
                  * (c - a) / (b - a)) * np.exp(c)
    expr2 = k * np.pi / (b - a) * np.sin(k * np.pi * 
                        (d - a) / (b - a))   - k * np.pi / (b - a) * np.sin(k 
                        * np.pi * (c - a) / (b - a)) * np.exp(c)
    chi = chi * (expr1 + expr2)
    
    value = {"chi":chi,"psi":psi }
    return value

def callPutOptionsPriceCoSMthd(cf, Cp, S0, r, tau, K, N, L):
    # cf   - characteristic function as a functon, in the book denoted as \varphi
    # CP   - C for call and P for put
    # S0   - Initial stock price
    # r    - interest rate (constant)
    # tau  - time to maturity
    # K    - list of strikes
    # N    - Number of expansion terms
    # L    - size of truncation domain (typ.:L=8 or L=10)

    # rehapse
    K=np.array(K).reshape([len(K),1])
    i=1j
    #turcation bounds
    x0=np.log(S0/K)
    a=0.0-L*np.sqrt(tau)
    b=0.0+L*np.sqrt(tau)
    #sumation from k=- to k=N-2
    k=np.linspace(0,N-1,N).reshape([N,1])
    u=k*np.pi/(b-a)
    #determing coiffesent for put price
    H_k=CallPutCoefficients(Cp,a,b,k)
    mat=np.exp(i*np.outer((x0-a),u))
    temp=cf(u)*H_k
    temp[0]=0.5*temp[0]
    value=np.exp(-r*tau)*K*np.real(mat@temp)
    return value

def BS_call_put_option_price(Cp, S_0, K, sigma, tau, r):
    cp=str(Cp).lower()
    K=np.array(K).reshape([len(K),1])
    d1=(np.log(S_0/K)+(r+0.5*sigma**2)*tau)/(sigma*np.sqrt(tau))
    d2    = d1 - sigma * np.sqrt(tau)
    if Cp == "c" or Cp == "1":
        value = stats.norm.cdf(d1) * S_0 - stats.norm.cdf(d2) * K * np.exp(-r * tau)
    elif Cp == "p" or Cp =="-1":
        value = stats.norm.cdf(-d2) * K * np.exp(-r * tau) - stats.norm.cdf(-d1)*S_0
    return value


# Streamlit app
st.set_page_config(page_title="European Option Pricing with COS Method", layout="wide")

st.title("European Option Pricing with COS Method")

# Create three columns for input
col1, col2, col3 = st.columns(3)

with col1:
    CP = st.radio("Option Type", ["Call", "Put"])
    S0 = st.number_input("Initial Stock Price (S0)", value=100.0, step=1.0)
    r = st.number_input("Risk-free Rate (r)", value=0.1, step=0.01, format="%.2f")

with col2:
    sigma = st.number_input("Volatility (σ)", value=0.25, step=0.01, format="%.2f")
    tau = st.number_input("Time to Maturity (τ) in years", value=0.1, step=0.1, format="%.1f")
    N = st.number_input("Number of Expansion Terms (N)", value=128, step=32)

with col3:
    L = st.number_input("Truncation Domain Size (L)", value=10, step=1)
    strike_prices = st.text_input("Enter strike prices (comma-separated)", "80, 90, 100, 110, 120")

# Convert strike prices to list
K = [float(k.strip()) for k in strike_prices.split(',')]

if st.button("Calculate Option Prices", key="calculate_button"):
    # Characteristic function for the Black-Scholes model
    cf = lambda u: np.exp((r - 0.5 * sigma**2) * 1j * u * tau - 0.5 * sigma**2 * u**2 * tau)
    
    # COS method calculation
    start_time = time.time()
    val_COS = callPutOptionsPriceCoSMthd(cf, CP[0].lower(), S0, r, tau, K, N, L)
    end_time = time.time()
    calculation_time = end_time - start_time
    
    # Black-Scholes calculation
    val_BS = BS_call_put_option_price(CP[0].lower(), S0, K, sigma, tau, r)
    
    # Plotting
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=K, y=val_COS.flatten(), mode='lines+markers', name='COS Price'))
    fig.add_trace(go.Scatter(x=K, y=val_BS.flatten(), mode='lines+markers', name='BS model', line=dict(dash='dash')))
    
    # Add vertical lines for each strike price
    for strike in K:
        fig.add_shape(
            type="line",
            x0=strike, y0=0, x1=strike, y1=max(val_COS.max(), val_BS.max()),
            line=dict(color="gray", width=1, dash="dot"),
        )
    
    fig.update_layout(
        title=f"{CP} Option Prices",
        xaxis_title="Strike Price (K)",
        yaxis_title="Option Price",
        legend_title="Pricing Method",
        hovermode="x unified",
        shapes=[dict(type="line", xref="paper", yref="paper", x0=0, y0=1.1, x1=1, y1=1.1, line_width=1)],
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.write(f"Calculation Time: {calculation_time:.4f} seconds")
    
    # Error calculation and display
    error = np.abs(val_COS - val_BS)
    
    # Create a DataFrame for results
    results_df = pd.DataFrame({
        "Strike Price": K,
        "COS Price": val_COS.flatten(),
        "BS Price": val_BS.flatten(),
        "Absolute Error": error.flatten()
    })
    
    st.write("Results:")
    st.dataframe(results_df.style.format({
        "Strike Price": "{:.2f}",
        "COS Price": "{:.6f}",
        "BS Price": "{:.6f}",
        "Absolute Error": "{:.2E}"
    }))

st.markdown("""
### Interpretation of Results
- The graph shows option prices calculated using both the COS method and the Black-Scholes model across different strike prices.
- Vertical dotted lines indicate the user-specified strike prices for easy reference.
- The COS method closely approximates the Black-Scholes prices, demonstrating its accuracy.
- The calculation time shows the efficiency of the COS method for pricing multiple options simultaneously.
- The table provides detailed results, including the absolute errors, which quantify the COS method's accuracy compared to the Black-Scholes model.

### Key Components of the COS Method:
1. Characteristic Function: Represents the Fourier transform of the probability density function.
2. Truncation Domain: Defines the integration range [a, b] for the approximation.
3. Fourier Cosine Expansion: Uses N terms to approximate the density function.
4. Coefficients: Calculated based on the payoff function of the option.

### Advantages of the COS Method:
1. Speed: Efficient for pricing multiple options with different strikes.
2. Accuracy: Provides results very close to analytical solutions (when available).
3. Flexibility: Can be adapted to various underlying asset price processes by changing the characteristic function.
""")