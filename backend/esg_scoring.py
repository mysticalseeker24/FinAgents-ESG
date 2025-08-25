"""
ESG scoring logic and calculations.

This module implements ESG scoring algorithms and methodologies
for evaluating companies based on environmental, social, and governance criteria.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
import json
import os
from datetime import datetime


# Sector ESG score mapping (predefined ESG scores by sector)
SECTOR_ESG_SCORES = {
    "Technology": 75,
    "Healthcare": 80,
    "Consumer Cyclical": 65,
    "Communication Services": 70,
    "Financial Services": 60,
    "Consumer Defensive": 75,
    "Industrials": 55,
    "Energy": 40,
    "Basic Materials": 45,
    "Real Estate": 70,
    "Utilities": 65,
    "Other": 50
}


def load_config() -> Dict[str, Any]:
    """Load configuration from settings.json file."""
    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "settings.json")
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        raise ValueError("Configuration file not found")
    except json.JSONDecodeError:
        raise ValueError("Invalid configuration file")


def get_sector_score(sector: str) -> float:
    """
    Get predefined ESG score for a sector.
    
    Args:
        sector: Company sector name
        
    Returns:
        ESG score for the sector (0-100)
    """
    return SECTOR_ESG_SCORES.get(sector, SECTOR_ESG_SCORES["Other"])


def calculate_volatility_score(history: pd.DataFrame, volatility_weight: float) -> float:
    """
    Calculate volatility score component.
    
    Args:
        history: Historical price data from yfinance
        volatility_weight: Weight for volatility in scoring
        
    Returns:
        Volatility score component
    """
    if history.empty or len(history) < 2:
        return 0.0
    
    # Calculate daily returns and volatility
    returns = history['Close'].pct_change().dropna()
    volatility = returns.std()
    
    # Convert to annualized volatility (assuming daily data)
    annualized_volatility = volatility * np.sqrt(252)
    
    # Volatility penalty: higher volatility = lower score
    volatility_score = -annualized_volatility * volatility_weight * 100
    
    return volatility_score


def calculate_market_cap_score(market_cap: float, market_cap_weight: float) -> float:
    """
    Calculate market cap score component.
    
    Args:
        market_cap: Company market capitalization
        market_cap_weight: Weight for market cap in scoring
        
    Returns:
        Market cap score component
    """
    if not market_cap or market_cap <= 0:
        return 0.0
    
    # Convert market cap to billions for scoring
    market_cap_billions = market_cap / 1e9
    
    # Score based on market cap: larger companies get higher scores
    # Cap at 10 points maximum
    market_cap_score = min(10.0, market_cap_billions * market_cap_weight)
    
    return market_cap_score


def calculate_esg_score(ticker: str) -> float:
    """
    Calculate ESG score for a given ticker.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        ESG score between 0 and 100
        
    Raises:
        ValueError: If ticker data cannot be retrieved
        Exception: For other calculation errors
    """
    try:
        # Load configuration
        config = load_config()
        weights = config.get("RISK_WEIGHTS", {})
        
        # Create yfinance ticker object
        yf_ticker = yf.Ticker(ticker)
        
        # Fetch historical data (1 year)
        history = yf_ticker.history(period="1y")
        if history.empty:
            raise ValueError(f"No historical data found for {ticker}")
        
        # Fetch company info
        info = yf_ticker.info
        if not info:
            raise ValueError(f"No company info found for {ticker}")
        
        # Extract key metrics
        market_cap = info.get('marketCap', 0)
        sector = info.get('sector', 'Other')
        
        # Base score
        base_score = 50.0
        
        # Sector score component
        sector_score = get_sector_score(sector)
        
        # Volatility score component
        volatility_weight = weights.get('volatility', 0.3)
        volatility_score = calculate_volatility_score(history, volatility_weight)
        
        # Market cap score component
        market_cap_weight = weights.get('market_cap', 0.2)
        market_cap_score = calculate_market_cap_score(market_cap, market_cap_weight)
        
        # Calculate final ESG score
        final_score = (
            base_score +
            sector_score * 0.3 +  # 30% weight for sector
            volatility_score +
            market_cap_score
        )
        
        # Clip score to [0, 100] range
        final_score = np.clip(final_score, 0.0, 100.0)
        
        return round(final_score, 2)
        
    except Exception as e:
        raise Exception(f"Failed to calculate ESG score for {ticker}: {str(e)}")


def calculate_detailed_esg_score(ticker: str) -> Dict[str, Any]:
    """
    Calculate detailed ESG score breakdown for a given ticker.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Dictionary with detailed ESG score components
    """
    try:
        # Load configuration
        config = load_config()
        weights = config.get("RISK_WEIGHTS", {})
        
        # Create yfinance ticker object
        yf_ticker = yf.Ticker(ticker)
        
        # Fetch historical data and info
        history = yf_ticker.history(period="1y")
        info = yf_ticker.info
        
        if history.empty or not info:
            raise ValueError(f"Insufficient data for {ticker}")
        
        # Extract metrics
        market_cap = info.get('marketCap', 0)
        sector = info.get('sector', 'Other')
        company_name = info.get('longName', ticker)
        
        # Calculate components
        base_score = 50.0
        sector_score = get_sector_score(sector)
        volatility_weight = weights.get('volatility', 0.3)
        volatility_score = calculate_volatility_score(history, volatility_weight)
        market_cap_weight = weights.get('market_cap', 0.2)
        market_cap_score = calculate_market_cap_score(market_cap, market_cap_weight)
        
        # Calculate final score
        final_score = (
            base_score +
            sector_score * 0.3 +
            volatility_score +
            market_cap_score
        )
        final_score = np.clip(final_score, 0.0, 100.0)
        
        # Calculate volatility metrics
        returns = history['Close'].pct_change().dropna()
        daily_volatility = returns.std()
        annualized_volatility = daily_volatility * np.sqrt(252)
        
        return {
            'ticker': ticker,
            'company_name': company_name,
            'sector': sector,
            'market_cap': market_cap,
            'final_esg_score': round(final_score, 2),
            'components': {
                'base_score': base_score,
                'sector_score': sector_score,
                'sector_weighted': sector_score * 0.3,
                'volatility_score': round(volatility_score, 2),
                'market_cap_score': round(market_cap_score, 2)
            },
            'metrics': {
                'daily_volatility': round(daily_volatility, 4),
                'annualized_volatility': round(annualized_volatility, 4),
                'market_cap_billions': round(market_cap / 1e9, 2)
            },
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        raise Exception(f"Failed to calculate detailed ESG score for {ticker}: {str(e)}")


def batch_calculate_esg_scores(tickers: list) -> Dict[str, float]:
    """
    Calculate ESG scores for multiple tickers.
    
    Args:
        tickers: List of stock ticker symbols
        
    Returns:
        Dictionary mapping tickers to ESG scores
    """
    results = {}
    
    for ticker in tickers:
        try:
            score = calculate_esg_score(ticker)
            results[ticker] = score
        except Exception as e:
            print(f"Warning: Failed to calculate ESG score for {ticker}: {e}")
            results[ticker] = None
    
    return results


def validate_esg_score(score: float) -> bool:
    """
    Validate that an ESG score is within valid range.
    
    Args:
        score: ESG score to validate
        
    Returns:
        True if score is valid, False otherwise
    """
    return isinstance(score, (int, float)) and 0 <= score <= 100


if __name__ == "__main__":
    # Example usage
    try:
        # Test with a well-known stock
        test_ticker = "AAPL"
        score = calculate_esg_score(test_ticker)
        print(f"ESG Score for {test_ticker}: {score}")
        
        # Get detailed breakdown
        detailed = calculate_detailed_esg_score(test_ticker)
        print(f"Detailed ESG Analysis for {test_ticker}:")
        print(json.dumps(detailed, indent=2))
        
    except Exception as e:
        print(f"Error: {e}")
