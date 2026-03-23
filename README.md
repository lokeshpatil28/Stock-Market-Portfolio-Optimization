# Portfolio Optimisation using Quadratic Programming

This project implements portfolio optimisation based on the Markowitz mean-variance framework using quadratic programming techniques from Operations Research. The objective is to determine optimal investment weights for a set of NSE stocks such that portfolio risk is minimised for a given target return.

Historical daily price data of selected stocks is collected using the yfinance library. Daily returns are computed from the price data and used to estimate the expected return vector and covariance matrix. These values are then used to formulate and solve a convex quadratic optimisation problem.

The optimisation model is solved using Karush-Kuhn-Tucker (KKT) optimality conditions and numerical linear algebra methods in Python. In addition to computing a single optimal portfolio, the model is also solved for multiple target return values to generate the efficient frontier, which illustrates the trade-off between portfolio risk and expected return.

## Project Structure

data/
- prices/                 raw stock price CSV files  
- mu.npy                  expected return vector  
- cov.npy                 covariance matrix  
- combined_prices.csv  

src/
- download_data.py        downloads stock data  
- compute_stats.py        computes returns and statistics  
- qp_solver.py            solves quadratic programming model  
- frontier.py             generates efficient frontier  
- plot_frontier.py        plots risk-return curve  

results/
- optimal_weights.csv  
- summary.csv  
- frontier.csv  
- frontier.png  

## How to Run

1. Download stock data  
python src/download_data.py  

2. Compute returns and covariance  
python src/compute_stats.py  

3. Solve portfolio optimisation  
cd src  
python qp_solver.py  

4. Generate efficient frontier  
python frontier.py  
python plot_frontier.py  

## Notes

- Long-only portfolio (no short selling allowed)  
- Full investment constraint is enforced  
- Historical returns are used as estimates of expected returns  
- This project is intended for academic learning and demonstration of quadratic programming in finance
