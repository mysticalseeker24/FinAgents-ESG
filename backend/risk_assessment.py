"""
Risk assessment and portfolio risk analysis.

This module implements risk assessment algorithms for evaluating
investment portfolios and individual securities.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import finnhub
from datetime import datetime, timedelta
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def calculate_returns(price_data: pd.Series) -> pd.Series:
    """
    Calculate daily returns from price data.
    
    Args:
        price_data: Series of closing prices
        
    Returns:
        Series of daily returns
    """
    return price_data.pct_change().dropna()


def calculate_volatility(returns: pd.Series, annualize: bool = True) -> float:
    """
    Calculate volatility from returns.
    
    Args:
        returns: Series of returns
        annualize: Whether to annualize the volatility
        
    Returns:
        Volatility measure
    """
    if returns.empty:
        return 0.0
    
    volatility = returns.std()
    
    if annualize:
        # Annualize assuming daily data (252 trading days)
        volatility = volatility * np.sqrt(252)
    
    return volatility


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """
    Calculate Sharpe ratio for a series of returns.
    
    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate (default: 2%)
        
    Returns:
        Sharpe ratio
    """
    if returns.empty:
        return 0.0
    
    # Calculate annualized return and volatility
    annual_return = returns.mean() * 252
    annual_volatility = returns.std() * np.sqrt(252)
    
    if annual_volatility == 0:
        return 0.0
    
    # Calculate Sharpe ratio
    sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility
    
    return sharpe_ratio


def calculate_correlation_matrix(stocks: List[str], finnhub_client: Optional[finnhub.Client] = None) -> pd.DataFrame:
    """
    Calculate pairwise correlation matrix for a list of stocks using Finnhub.
    
    Args:
        stocks: List of stock symbols
        finnhub_client: Optional Finnhub client instance
        
    Returns:
        Correlation matrix DataFrame
    """
    # Fetch historical data for all stocks
    stock_data = {}
    
    # Create Finnhub client if not provided
    if finnhub_client is None:
        import os
        import json
        config_path = os.path.join(os.path.dirname(__file__), "..", "config", "settings.json")
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            api_key = config.get("FINNHUB_API_KEY")
            if not api_key:
                raise ValueError("FINNHUB_API_KEY not found in configuration")
            finnhub_client = finnhub.Client(api_key=api_key)
        except Exception as e:
            raise ValueError(f"Failed to initialize Finnhub client: {e}")
    
    for symbol in stocks:
        try:
            # Get 1 year of daily data
            from_timestamp = int((datetime.now() - timedelta(days=365)).timestamp())
            to_timestamp = int(datetime.now().timestamp())
            
            candles = finnhub_client.stock_candles(symbol, 'D', from_timestamp, to_timestamp)
            if candles and 'c' in candles and candles['c']:
                stock_data[symbol] = candles['c']
        except Exception as e:
            print(f"Warning: Failed to fetch data for {symbol}: {e}")
            continue
    
    if not stock_data:
        raise ValueError("No stock data could be retrieved")
    
    # Create DataFrame with all stock prices
    price_df = pd.DataFrame(stock_data)
    
    # Calculate returns
    returns_df = price_df.pct_change().dropna()
    
    # Calculate correlation matrix
    correlation_matrix = returns_df.corr()
    
    return correlation_matrix


def calculate_portfolio_variance(weights: List[float], returns_df: pd.DataFrame) -> float:
    """
    Calculate portfolio variance using the covariance matrix.
    
    Args:
        weights: Portfolio weights for each asset
        returns_df: DataFrame of returns for all assets
        
    Returns:
        Portfolio variance
    """
    if len(weights) != len(returns_df.columns):
        raise ValueError("Number of weights must match number of assets")
    
    # Calculate covariance matrix
    cov_matrix = returns_df.cov() * 252  # Annualize
    
    # Calculate portfolio variance: w^T * Σ * w
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    
    return portfolio_variance


def calculate_portfolio_volatility(weights: List[float], returns_df: pd.DataFrame) -> float:
    """
    Calculate portfolio volatility.
    
    Args:
        weights: Portfolio weights for each asset
        returns_df: DataFrame of returns for all assets
        
    Returns:
        Portfolio volatility
    """
    portfolio_variance = calculate_portfolio_variance(weights, returns_df)
    return np.sqrt(portfolio_variance)


def calculate_portfolio_sharpe_ratio(weights: List[float], returns_df: pd.DataFrame, 
                                   risk_free_rate: float = 0.02) -> float:
    """
    Calculate portfolio Sharpe ratio.
    
    Args:
        weights: Portfolio weights for each asset
        returns_df: DataFrame of returns for all assets
        risk_free_rate: Annual risk-free rate
        
    Returns:
        Portfolio Sharpe ratio
    """
    # Calculate portfolio return
    asset_returns = returns_df.mean() * 252  # Annualized returns
    portfolio_return = np.dot(weights, asset_returns)
    
    # Calculate portfolio volatility
    portfolio_volatility = calculate_portfolio_volatility(weights, returns_df)
    
    if portfolio_volatility == 0:
        return 0.0
    
    # Calculate Sharpe ratio
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    
    return sharpe_ratio


def assess_portfolio(stocks: List[str]) -> Dict[str, Any]:
    """
    Assess portfolio risk for a list of stocks.
    
    Args:
        stocks: List of stock ticker symbols
        
    Returns:
        Dictionary containing risk metrics
    """
    try:
        if not stocks:
            raise ValueError("Stock list cannot be empty")
        
        # Fetch historical data for all stocks using Finnhub
        stock_data = {}
        
        # Initialize Finnhub client
        import os
        import json
        config_path = os.path.join(os.path.dirname(__file__), "..", "config", "settings.json")
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            api_key = config.get("FINNHUB_API_KEY")
            if not api_key:
                raise ValueError("FINNHUB_API_KEY not found in configuration")
            finnhub_client = finnhub.Client(api_key=api_key)
        except Exception as e:
            raise ValueError(f"Failed to initialize Finnhub client: {e}")
        
        for symbol in stocks:
            try:
                # Get 1 year of daily data
                from_timestamp = int((datetime.now() - timedelta(days=365)).timestamp())
                to_timestamp = int(datetime.now().timestamp())
                
                candles = finnhub_client.stock_candles(symbol, 'D', from_timestamp, to_timestamp)
                if candles and 'c' in candles and candles['c']:
                    stock_data[symbol] = candles['c']
            except Exception as e:
                print(f"Warning: Failed to fetch data for {symbol}: {e}")
                continue
        
        if not stock_data:
            raise ValueError("No stock data could be retrieved")
        
        # Create DataFrame with all stock prices
        price_df = pd.DataFrame(stock_data)
        
        # Calculate returns
        returns_df = price_df.pct_change().dropna()
        
        # Calculate correlation matrix
        correlation_matrix = returns_df.corr()
        
        # Calculate individual stock metrics
        individual_metrics = {}
        for symbol in stocks:
            if symbol in returns_df.columns:
                returns = returns_df[symbol]
                individual_metrics[symbol] = {
                    'volatility': calculate_volatility(returns),
                    'sharpe_ratio': calculate_sharpe_ratio(returns),
                    'annual_return': returns.mean() * 252
                }
        
        # Calculate portfolio metrics (equal weight for simplicity)
        n_stocks = len(stocks)
        weights = [1.0 / n_stocks] * n_stocks
        
        portfolio_variance = calculate_portfolio_variance(weights, returns_df)
        portfolio_volatility = calculate_portfolio_volatility(weights, returns_df)
        portfolio_sharpe = calculate_portfolio_sharpe_ratio(weights, returns_df)
        
        # Calculate sector concentration (if sector info available)
        sector_concentration = calculate_sector_concentration(stocks)
        
        # Calculate ESG risk factors
        esg_risk_factors = calculate_esg_risk_factors(stocks)
        
        return {
            'portfolio_metrics': {
                'variance': round(portfolio_variance, 6),
                'volatility': round(portfolio_volatility, 4),
                'sharpe_ratio': round(portfolio_sharpe, 4),
                'number_of_stocks': n_stocks
            },
            'individual_metrics': individual_metrics,
            'correlation_matrix': correlation_matrix.round(4).to_dict(),
            'sector_concentration': sector_concentration,
            'esg_risk_factors': esg_risk_factors,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        raise Exception(f"Failed to assess portfolio risk: {str(e)}")


def calculate_sector_concentration(stocks: List[str]) -> Dict[str, Any]:
    """
    Calculate sector concentration for a portfolio.
    
    Args:
        stocks: List of stock tickers
        
    Returns:
        Dictionary with sector concentration metrics
    """
    sector_counts = {}
    sector_market_caps = {}
    
    for ticker in stocks:
        try:
            yf_ticker = yf.Ticker(ticker)
            info = yf_ticker.info
            
            sector = info.get('sector', 'Unknown')
            market_cap = info.get('marketCap', 0)
            
            # Count stocks by sector
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
            
            # Sum market caps by sector
            sector_market_caps[sector] = sector_market_caps.get(sector, 0) + market_cap
            
        except Exception:
            continue
    
    # Calculate concentration metrics
    total_stocks = len(stocks)
    total_market_cap = sum(sector_market_caps.values())
    
    sector_concentration = {
        'sector_counts': sector_counts,
        'sector_weights': {sector: count/total_stocks for sector, count in sector_counts.items()},
        'sector_market_caps': sector_market_caps,
        'sector_market_cap_weights': {sector: mcap/total_market_cap for sector, mcap in sector_market_caps.items()},
        'herfindahl_index': sum((count/total_stocks)**2 for count in sector_counts.values())
    }
    
    return sector_concentration


def calculate_esg_risk_factors(stocks: List[str]) -> Dict[str, Any]:
    """
    Calculate ESG risk factors for a portfolio.
    
    Args:
        stocks: List of stock tickers
        
    Returns:
        Dictionary with ESG risk metrics
    """
    # This is a placeholder - in a real implementation, you would
    # integrate with ESG data providers or use the ESG scoring module
    esg_risk_factors = {
        'high_risk_sectors': ['Energy', 'Basic Materials'],
        'esg_score_range': 'To be calculated',
        'sustainability_rating': 'To be calculated',
        'climate_risk_exposure': 'To be calculated'
    }
    
    return esg_risk_factors


def calculate_var(returns: pd.Series, confidence_level: float = 0.05) -> float:
    """
    Calculate Value at Risk (VaR) for a given confidence level.
    
    Args:
        returns: Series of returns
        confidence_level: Confidence level for VaR (default: 5%)
        
    Returns:
        VaR value
    """
    if returns.empty:
        return 0.0
    
    # Calculate VaR using historical simulation
    var = np.percentile(returns, confidence_level * 100)
    
    # Annualize if needed
    var_annualized = var * np.sqrt(252)
    
    return var_annualized


def calculate_max_drawdown(price_data: pd.Series) -> float:
    """
    Calculate maximum drawdown from peak.
    
    Args:
        price_data: Series of prices
        
    Returns:
        Maximum drawdown as a percentage
    """
    if price_data.empty:
        return 0.0
    
    # Calculate cumulative returns
    cumulative_returns = (1 + price_data.pct_change()).cumprod()
    
    # Calculate running maximum
    running_max = cumulative_returns.expanding().max()
    
    # Calculate drawdown
    drawdown = (cumulative_returns - running_max) / running_max
    
    # Get maximum drawdown
    max_drawdown = drawdown.min()
    
    return max_drawdown


if __name__ == "__main__":
    # Example usage
    try:
        # Test with some well-known stocks
        test_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        
        print("Assessing portfolio risk...")
        risk_assessment = assess_portfolio(test_stocks)
        
        print("\nPortfolio Risk Assessment Results:")
        print("=" * 50)
        
        print(f"Portfolio Variance: {risk_assessment['portfolio_metrics']['variance']}")
        print(f"Portfolio Volatility: {risk_assessment['portfolio_metrics']['volatility']}")
        print(f"Portfolio Sharpe Ratio: {risk_assessment['portfolio_metrics']['sharpe_ratio']}")
        print(f"Number of Stocks: {risk_assessment['portfolio_metrics']['number_of_stocks']}")
        
        print("\nIndividual Stock Metrics:")
        for ticker, metrics in risk_assessment['individual_metrics'].items():
            print(f"{ticker}: Vol={metrics['volatility']:.4f}, "
                  f"Sharpe={metrics['sharpe_ratio']:.4f}, "
                  f"Return={metrics['annual_return']:.4f}")
        
        print("\nSector Concentration:")
        for sector, weight in risk_assessment['sector_concentration']['sector_weights'].items():
            print(f"{sector}: {weight:.2%}")
        
    except Exception as e:
        print(f"Error: {e}")
