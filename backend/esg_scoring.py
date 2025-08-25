"""
ESG scoring logic and calculations using Finnhub API.

This module implements ESG scoring algorithms and methodologies
for evaluating companies based on environmental, social, and governance criteria.
Uses Finnhub API for reliable market data with proper rate limiting and webhook support.
"""

import finnhub
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import json
import os
import time
import logging
from datetime import datetime, timedelta
import requests
from fastapi import HTTPException

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Market-specific ticker lists for Finnhub
MARKET_TICKERS = {
    "US": {
        "Technology": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "NFLX", "ADBE", "CRM"],
        "Healthcare": ["JNJ", "PFE", "UNH", "ABBV", "TMO", "ABT", "LLY", "DHR", "BMY", "AMGN"],
        "Financial Services": ["JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "AXP", "CB", "PGR"],
        "Consumer": ["PG", "KO", "PEP", "WMT", "HD", "MCD", "DIS", "NKE", "SBUX", "TGT"],
        "Industrials": ["BA", "CAT", "GE", "MMM", "HON", "UPS", "FDX", "LMT", "RTX", "DE"]
    },
    "EUROPE": {
        "Technology": ["ASML", "SAP", "NOVO", "NESTLE", "ROCHE", "NOVARTIS", "ASML", "SAP"],
        "Healthcare": ["NOVO", "ROCHE", "NOVARTIS", "SANOFI", "GSK", "ASTRAZENECA"],
        "Financial Services": ["HSBC", "BNP", "DEUTSCHE", "UBS", "CREDIT_SUISSE", "BARCLAYS"],
        "Consumer": ["NESTLE", "UNILEVER", "LVMH", "HERMES", "RICHEMONT"],
        "Industrials": ["SIEMENS", "VOLKSWAGEN", "BMW", "DAIMLER", "AIRBUS"]
    },
    "ASIA": {
        "Technology": ["TSM", "BABA", "TENCENT", "SOFTBANK", "SONY", "CANON", "HITACHI"],
        "Healthcare": ["TMO", "ILLUMINA", "GILEAD", "AMGEN", "BIOGEN"],
        "Financial Services": ["MITSUBISHI_UFJ", "SUMITOMO_MITSUI", "NOMURA", "DBS", "OCBC"],
        "Consumer": ["TOYOTA", "HONDA", "NINTENDO", "UNIQLO", "SEVEN_AND_I"],
        "Industrials": ["MITSUBISHI", "HITACHI", "PANASONIC", "SHARP", "TOSHIBA"]
    }
}


class FinnhubClient:
    """Finnhub API client with rate limiting and error handling."""
    
    def __init__(self, api_key: str, config: Dict[str, Any]):
        self.client = finnhub.Client(api_key=api_key)
        self.requests_per_minute = config.get("requests_per_minute", 20)
        self.delay_between_requests = config.get("delay_between_requests", 3.0)
        self.max_retries = config.get("max_retries", 3)
        self.retry_delay = config.get("retry_delay", 2.0)
        self.batch_size = config.get("batch_size", 20)
        
        self.last_request_time = 0
        self.request_count = 0
        self.minute_start = time.time()
    
    def wait_if_needed(self):
        """Wait if rate limit is reached."""
        current_time = time.time()
        
        # Reset counter if a minute has passed
        if current_time - self.minute_start >= 60:
            self.request_count = 0
            self.minute_start = current_time
        
        # Check if we've hit the rate limit
        if self.request_count >= self.requests_per_minute:
            wait_time = 60 - (current_time - self.minute_start)
            if wait_time > 0:
                logger.info(f"Rate limit reached, waiting {wait_time:.1f} seconds")
                time.sleep(wait_time)
                self.request_count = 0
                self.minute_start = time.time()
        
        # Wait between requests
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.delay_between_requests:
            sleep_time = self.delay_between_requests - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
        self.request_count += 1
    
    def get_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get real-time quote for a symbol."""
        try:
            self.wait_if_needed()
            quote = self.client.quote(symbol)
            if quote and 'c' in quote:  # Check if we have current price
                return quote
            return None
        except Exception as e:
            logger.error(f"Error fetching quote for {symbol}: {e}")
            return None
    
    def get_company_profile(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get company profile information."""
        try:
            self.wait_if_needed()
            profile = self.client.company_profile2(symbol=symbol)
            if profile and isinstance(profile, dict):
                return profile
            return None
        except Exception as e:
            logger.error(f"Error fetching profile for {symbol}: {e}")
            return None
    
    def get_financial_ratios(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get financial ratios for a symbol."""
        try:
            self.wait_if_needed()
            ratios = self.client.financial_ratios(symbol, 'annual')
            if ratios and isinstance(ratios, dict):
                return ratios
            return None
        except Exception as e:
            logger.error(f"Error fetching ratios for {symbol}: {e}")
            return None
    
    def get_stock_symbols(self, exchange: str = 'US') -> List[Dict[str, Any]]:
        """Get stock symbols for a specific exchange."""
        try:
            self.wait_if_needed()
            symbols = self.client.stock_symbols(exchange)
            if symbols and isinstance(symbols, list):
                return symbols
            return []
        except Exception as e:
            logger.error(f"Error fetching symbols for {exchange}: {e}")
            return []
    
    def get_candles(self, symbol: str, resolution: str = 'D', 
                    from_timestamp: Optional[int] = None, 
                    to_timestamp: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Get candlestick data for a symbol."""
        try:
            self.wait_if_needed()
            
            # Default to last 1 year if no timestamps provided
            if from_timestamp is None:
                from_timestamp = int((datetime.now() - timedelta(days=365)).timestamp())
            if to_timestamp is None:
                to_timestamp = int(datetime.now().timestamp())
            
            candles = self.client.stock_candles(symbol, resolution, from_timestamp, to_timestamp)
            if candles and isinstance(candles, dict) and 'c' in candles:
                return candles
            return None
        except Exception as e:
            logger.error(f"Error fetching candles for {symbol}: {e}")
            return None


class MarketSelector:
    """Market and stock selection based on configuration."""
    
    def __init__(self, config: Dict[str, Any]):
        self.target_markets = config.get("target_markets", ["US"])
        self.target_sectors = config.get("target_sectors", ["Technology", "Healthcare"])
        self.excluded_sectors = config.get("excluded_sectors", ["Energy", "Basic Materials"])
        self.market_cap_range = config.get("market_cap_range", {"min_billions": 1.0, "max_billions": 1000.0})
        self.stock_types = config.get("stock_types", ["common"])
        self.geographic_focus = config.get("geographic_focus", "US")
    
    def get_filtered_tickers(self) -> List[str]:
        """Get filtered tickers based on market selection criteria."""
        all_tickers = []
        
        for market in self.target_markets:
            if market in MARKET_TICKERS:
                for sector in self.target_sectors:
                    if sector in MARKET_TICKERS[market]:
                        all_tickers.extend(MARKET_TICKERS[market][sector])
        
        # Remove duplicates
        all_tickers = list(set(all_tickers))
        
        logger.info(f"Selected {len(all_tickers)} tickers from {len(self.target_markets)} markets and {len(self.target_sectors)} sectors")
        return all_tickers
    
    def filter_by_market_cap(self, ticker: str, market_cap: float) -> bool:
        """Filter ticker by market cap range."""
        if not market_cap or market_cap <= 0:
            return False
        
        market_cap_billions = market_cap / 1e9
        return (self.market_cap_range["min_billions"] <= market_cap_billions <= 
                self.market_cap_range["max_billions"])


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


def calculate_volatility_score(candles: Dict[str, Any], volatility_weight: float) -> float:
    """
    Calculate volatility score component from Finnhub candles data.
    
    Args:
        candles: Candlestick data from Finnhub
        volatility_weight: Weight for volatility in scoring
        
    Returns:
        Volatility score component
    """
    if not candles or 'c' not in candles or not candles['c']:
        return 0.0
    
    # Extract close prices
    close_prices = candles['c']
    if len(close_prices) < 2:
        return 0.0
    
    # Calculate daily returns and volatility
    returns = pd.Series(close_prices).pct_change().dropna()
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


def calculate_esg_score(symbol: str, finnhub_client: Optional[FinnhubClient] = None) -> float:
    """
    Calculate ESG score for a given symbol using Finnhub data.
    
    Args:
        symbol: Stock symbol
        finnhub_client: Optional Finnhub client instance
        
    Returns:
        ESG score between 0 and 100
        
    Raises:
        ValueError: If symbol data cannot be retrieved
        Exception: For other calculation errors
    """
    try:
        # Load configuration
        config = load_config()
        weights = config.get("RISK_WEIGHTS", {})
        
        # Create Finnhub client if not provided
        if finnhub_client is None:
            api_key = config.get("FINNHUB_API_KEY")
            if not api_key:
                raise ValueError("FINNHUB_API_KEY not found in configuration")
            finnhub_client = FinnhubClient(api_key, config.get("FINNHUB_CONFIG", {}))
        
        # Get company profile
        profile = finnhub_client.get_company_profile(symbol)
        if not profile:
            raise ValueError(f"Could not retrieve profile for {symbol}")
        
        # Get quote data
        quote = finnhub_client.get_quote(symbol)
        if not quote:
            raise ValueError(f"Could not retrieve quote for {symbol}")
        
        # Get historical data for volatility calculation
        candles = finnhub_client.get_candles(symbol, 'D')
        if not candles:
            raise ValueError(f"Could not retrieve historical data for {symbol}")
        
        # Extract key metrics
        market_cap = profile.get('marketCapitalization', 0)
        sector = profile.get('finnhubIndustry', 'Other')
        
        # Base score
        base_score = 50.0
        
        # Sector score component
        sector_score = get_sector_score(sector)
        
        # Volatility score component
        volatility_weight = weights.get('volatility', 0.3)
        volatility_score = calculate_volatility_score(candles, volatility_weight)
        
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
        raise Exception(f"Failed to calculate ESG score for {symbol}: {str(e)}")


def calculate_detailed_esg_score(symbol: str, finnhub_client: Optional[FinnhubClient] = None) -> Dict[str, Any]:
    """
    Calculate detailed ESG score breakdown for a given symbol.
    
    Args:
        symbol: Stock symbol
        finnhub_client: Optional Finnhub client instance
        
    Returns:
        Dictionary with detailed ESG score components
    """
    try:
        # Load configuration
        config = load_config()
        weights = config.get("RISK_WEIGHTS", {})
        
        # Create Finnhub client if not provided
        if finnhub_client is None:
            api_key = config.get("FINNHUB_API_KEY")
            if not api_key:
                raise ValueError("FINNHUB_API_KEY not found in configuration")
            finnhub_client = FinnhubClient(api_key, config.get("FINNHUB_CONFIG", {}))
        
        # Get company profile
        profile = finnhub_client.get_company_profile(symbol)
        if not profile:
            raise ValueError(f"Could not retrieve profile for {symbol}")
        
        # Get quote data
        quote = finnhub_client.get_quote(symbol)
        if not quote:
            raise ValueError(f"Could not retrieve quote for {symbol}")
        
        # Get historical data
        candles = finnhub_client.get_candles(symbol, 'D')
        if not candles:
            raise ValueError(f"Could not retrieve historical data for {symbol}")
        
        # Extract metrics
        market_cap = profile.get('marketCapitalization', 0)
        sector = profile.get('finnhubIndustry', 'Other')
        company_name = profile.get('name', symbol)
        
        # Calculate components
        base_score = 50.0
        sector_score = get_sector_score(sector)
        volatility_weight = weights.get('volatility', 0.3)
        volatility_score = calculate_volatility_score(candles, volatility_weight)
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
        
        # Calculate volatility metrics from candles
        close_prices = candles['c']
        returns = pd.Series(close_prices).pct_change().dropna()
        daily_volatility = returns.std()
        annualized_volatility = daily_volatility * np.sqrt(252)
        
        return {
            'symbol': symbol,
            'company_name': company_name,
            'sector': sector,
            'market_cap': market_cap,
            'current_price': quote.get('c', 0),
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
        raise Exception(f"Failed to calculate detailed ESG score for {symbol}: {str(e)}")


def batch_calculate_esg_scores(symbols: List[str], config: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    """
    Calculate ESG scores for multiple symbols with proper rate limiting.
    
    Args:
        symbols: List of stock symbols
        config: Optional configuration dictionary
        
    Returns:
        Dictionary mapping symbols to ESG scores
    """
    if config is None:
        config = load_config()
    
    api_key = config.get("FINNHUB_API_KEY")
    if not api_key:
        raise ValueError("FINNHUB_API_KEY not found in configuration")
    
    finnhub_client = FinnhubClient(api_key, config.get("FINNHUB_CONFIG", {}))
    batch_size = config.get("FINNHUB_CONFIG", {}).get("batch_size", 20)
    
    results = {}
    
    # Process symbols in batches
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i + batch_size]
        logger.info(f"Processing batch {i//batch_size + 1}/{(len(symbols) + batch_size - 1)//batch_size} ({len(batch)} symbols)")
        
        for symbol in batch:
            try:
                logger.info(f"Processing {symbol}...")
                score = calculate_esg_score(symbol, finnhub_client)
                results[symbol] = score
                
            except Exception as e:
                logger.warning(f"Failed to calculate ESG score for {symbol}: {e}")
                results[symbol] = None
        
        # Wait between batches to avoid overwhelming the API
        if i + batch_size < len(symbols):
            wait_time = 5.0  # Wait 5 seconds between batches
            logger.info(f"Waiting {wait_time}s before next batch...")
            time.sleep(wait_time)
    
    return results


def get_market_symbols(market: str, sector: Optional[str] = None) -> List[str]:
    """
    Get symbols for a specific market and optionally sector.
    
    Args:
        market: Market name (US, EUROPE, ASIA)
        sector: Optional sector name
        
    Returns:
        List of symbol strings
    """
    if market not in MARKET_TICKERS:
        return []
    
    if sector:
        return MARKET_TICKERS[market].get(sector, [])
    else:
        all_symbols = []
        for sector_symbols in MARKET_TICKERS[market].values():
            all_symbols.extend(sector_symbols)
        return all_symbols


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
        # Load configuration
        config = load_config()
        
        # Initialize market selector
        market_selector = MarketSelector(config.get("MARKET_SELECTION", {}))
        
        # Get filtered symbols
        symbols = market_selector.get_filtered_tickers()
        print(f"Selected {len(symbols)} symbols for analysis")
        
        # Test with a small batch
        test_symbols = symbols[:5]
        print(f"Testing with symbols: {test_symbols}")
        
        # Calculate ESG scores with rate limiting
        scores = batch_calculate_esg_scores(test_symbols, config)
        
        print("\nESG Scores:")
        for symbol, score in scores.items():
            if score is not None:
                print(f"{symbol}: {score}")
            else:
                print(f"{symbol}: Failed")
        
    except Exception as e:
        print(f"Error: {e}")
