# Quanto Options Analysis: Gold USD/PLN Cross-Currency Derivatives

## Project Overview

This project provides a comprehensive analysis of quanto options, focusing on gold (USD-denominated) with PLN payouts. The implementation covers mathematical modeling, calibration, pricing, and hedging strategies for cross-currency derivatives using correlated geometric Brownian motion models.

## Instrument Specification

### Quanto Option Details
- **Underlying Asset**: Gold (London Gold Fixing Price in USD)
- **Currency Exposure**: USD/PLN exchange rate
- **Payoff Structure**: `100 PLN × max(S(T)/S(0) - K, 0)`
- **Maturity**: June 30, 2024
- **Analysis Period**: June 30, 2023 - June 30, 2024
- **Strike Range**: 0 ≤ K ≤ 2

### Key Characteristics
- **Cross-Currency Risk**: Gold priced in USD, payoff in PLN
- **Correlation Effects**: Correlated movements between gold prices and FX rates
- **Dual Modeling Approaches**: USD-based vs PLN-based formulations

## Technical Implementation

### Mathematical Framework

#### Model 1: USD Gold + USD/PLN Process
- **S(t)**: Gold price in USD (Geometric Brownian Motion)
- **X(t)**: USD/PLN exchange rate (Geometric Brownian Motion)
- **Correlation**: ρ between dS and dX processes

#### Model 2: PLN Gold + USD/PLN Process  
- **Y(t) = X(t)·S(t)**: Gold price in PLN
- **X(t)**: USD/PLN exchange rate
- **Transformed Dynamics**: Derived PLN-denominated processes

### Core Features

1. **Parameter Calibration**
   - Historical data analysis for gold prices and FX rates
   - Interest rate incorporation (USD and PLN rates)
   - Correlation estimation between underlying processes
   - Model validation and goodness-of-fit testing

2. **Pricing Methodology**
   - **Analytical Solutions**: Closed-form quanto option formulas
   - **Risk-Neutral Valuation**: Appropriate measure changes
   - **Cross-Currency Adjustments**: Quanto drift corrections
   - **Dual Approach Comparison**: USD vs PLN formulation consistency

3. **Hedging Portfolio Construction**
   - **Delta Hedging**: Sensitivities to underlying assets
   - **Currency Hedging**: FX exposure management  
   - **Dynamic Rebalancing**: Optimal rehedging frequencies
   - **Portfolio Composition Analysis**: Asset allocation insights

## Interactive Visualizations

The project features **advanced interactive charts built with Plotly.go architecture**:

### Pricing Analysis
- **3D Price Surfaces**: Option value vs gold price and FX rate
- **Strike Sensitivity Charts**: Price behavior across different K values
- **Time Decay Visualization**: Theta effects over option lifetime
- **Correlation Impact Heatmaps**: Price sensitivity to ρ parameter

### Hedging Analysis  
- **Dynamic Hedge Ratios**: Real-time portfolio composition
- **Rehedging Frequency Studies**: P&L impact of trading frequency
- **Historical Backtesting**: Performance on actual market data
- **Portfolio Risk Decomposition**: Component-wise risk attribution

### Comparative Analysis
- **Model Convergence Studies**: USD vs PLN approach comparison
- **Payoff Structure Comparisons**: Traditional vs quanto option profiles
- **Cross-Currency Risk Metrics**: Exposure quantification

## Research Components

### 1. Calibration Framework
- **Historical Data Integration**: Gold prices and USD/PLN rates
- **Parameter Estimation**: Maximum likelihood and method of moments
- **Model Selection**: Statistical testing for optimal specification
- **Robustness Analysis**: Parameter stability over time

### 2. Pricing Engine
- **Analytical Solutions**: Exact quanto option formulas
- **Numerical Validation**: Monte Carlo cross-verification
- **Sensitivity Analysis**: Greeks computation and interpretation
- **Model Arbitrage Testing**: No-arbitrage condition verification

### 3. Hedging Simulation
- **Dynamic Hedging**: Continuous-time approximation strategies
- **Transaction Cost Analysis**: Realistic trading friction incorporation
- **Performance Metrics**: Sharpe ratios, maximum drawdown, hit rates
- **Frequency Optimization**: Optimal rebalancing intervals

### 4. Comparative Studies
- **Cross-Model Validation**: Consistency between formulations
- **Traditional vs Quanto**: Standard European vs cross-currency options
- **Perspective Analysis**: Polish vs American investor viewpoints

## Key Findings

### Model Insights
- **Correlation Effects**: Significant impact on quanto option pricing
- **Currency Risk Premium**: Quantification of cross-currency exposure
- **Hedging Complexity**: Multi-asset portfolio requirements
- **Formulation Equivalence**: Theoretical consistency validation

### Practical Applications
- **Portfolio Management**: Cross-currency risk management tools
- **Arbitrage Detection**: Model-based trading opportunities
- **Risk Assessment**: Comprehensive exposure measurement
- **Hedging Optimization**: Cost-effective protection strategies


## Technical Applications

### For Quantitative Analysts
- **Cross-Currency Modeling**: Advanced correlation structures
- **Hedging Strategy Development**: Multi-asset portfolio optimization
- **Risk Model Validation**: Cross-model consistency testing

### For Risk Managers  
- **Currency Exposure Measurement**: Comprehensive risk metrics
- **Hedging Cost Analysis**: Optimal rebalancing frequencies
- **Scenario Testing**: Stress testing under extreme correlations

### For Portfolio Managers
- **Cross-Border Investments**: Currency-hedged exposure strategies
- **Structured Products**: Quanto-based product development
- **Performance Attribution**: Risk-return decomposition

## Validation & Testing

- **Theoretical Consistency**: Mathematical equivalence verification
- **Historical Backtesting**: Real market data performance
- **Monte Carlo Validation**: Numerical accuracy confirmation
- **Sensitivity Robustness**: Parameter stability analysis

## Results Highlights

The interactive visualizations reveal:
- **Non-linear correlation effects** on option pricing
- **Complex hedging dynamics** requiring multi-asset strategies  
- **Frequency-dependent hedging performance** with optimal rebalancing intervals
- **Model formulation equivalence** under ideal geometric Brownian motion assumptions

All charts provide full interactivity for parameter exploration and scenario analysis.

---

*This project demonstrates advanced quantitative finance techniques in cross-currency derivatives, combining rigorous mathematical modeling with practical implementation and comprehensive visualization tools.
Project was developed as part of Financial Engineering I course (2024/2025), demonstrating advanced numerical methods and interactive data visualization in quantitative finance.*
