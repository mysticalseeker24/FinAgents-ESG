"""
In-memory data storage for the ESG Investment Advisor.

This module provides a singleton DataStore class that manages all application data
in memory, including user preferences, ESG scores, market data, and recommendations.
"""

import pandas as pd
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import json


@dataclass
class UserPreferences:
    """User investment preferences and constraints."""
    
    risk_tolerance: str = "moderate"  # low, moderate, high
    investment_horizon: str = "medium"  # short, medium, long
    esg_priority: str = "balanced"  # environmental, social, governance, balanced
    sector_preferences: List[str] = field(default_factory=list)
    excluded_sectors: List[str] = field(default_factory=list)
    min_investment: float = 1000.0
    max_investment: float = 100000.0


@dataclass
class ESGScore:
    """ESG scoring data for a company."""
    
    ticker: str
    company_name: str
    environmental_score: float
    social_score: float
    governance_score: float
    overall_esg_score: float
    sector: str
    market_cap: float
    last_updated: str


@dataclass
class MarketData:
    """Market data for a company."""
    
    ticker: str
    current_price: float
    volume: int
    market_cap: float
    pe_ratio: float
    dividend_yield: float
    beta: float
    volatility: float
    last_updated: str


@dataclass
class Recommendation:
    """Investment recommendation for a company."""
    
    ticker: str
    company_name: str
    recommendation: str  # buy, hold, sell
    confidence_score: float
    esg_score: float
    risk_score: float
    target_price: float
    reasoning: str
    timestamp: str


class DataStore:
    """
    Singleton class for in-memory data storage.
    
    Manages all application data including user preferences, ESG scores,
    market data, and investment recommendations.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DataStore, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        # User preferences storage
        self.user_preferences: Dict[str, UserPreferences] = {}
        
        # ESG scores storage
        self.esg_scores: pd.DataFrame = pd.DataFrame(columns=[
            'ticker', 'company_name', 'environmental_score', 'social_score',
            'governance_score', 'overall_esg_score', 'sector', 'market_cap', 'last_updated'
        ])
        
        # Market data storage
        self.market_data: pd.DataFrame = pd.DataFrame(columns=[
            'ticker', 'current_price', 'volume', 'market_cap', 'pe_ratio',
            'dividend_yield', 'beta', 'volatility', 'last_updated'
        ])
        
        # Recommendations storage
        self.recommendations: Dict[str, List[Recommendation]] = {}
        
        # Risk assessment results
        self.risk_results: Dict[str, Dict[str, Any]] = {}
        
        # Portfolio data
        self.portfolios: Dict[str, Dict[str, Any]] = {}
        
        self._initialized = True
    
    def add_user_preferences(self, user_id: str, preferences: UserPreferences) -> None:
        """Add or update user preferences."""
        self.user_preferences[user_id] = preferences
    
    def get_user_preferences(self, user_id: str) -> Optional[UserPreferences]:
        """Get user preferences by user ID."""
        return self.user_preferences.get(user_id)
    
    def add_esg_score(self, esg_score: ESGScore) -> None:
        """Add or update ESG score for a company."""
        # Remove existing entry if it exists
        self.esg_scores = self.esg_scores[self.esg_scores['ticker'] != esg_score.ticker]
        
        # Add new entry
        new_row = pd.DataFrame([{
            'ticker': esg_score.ticker,
            'company_name': esg_score.company_name,
            'environmental_score': esg_score.environmental_score,
            'social_score': esg_score.social_score,
            'governance_score': esg_score.governance_score,
            'overall_esg_score': esg_score.overall_esg_score,
            'sector': esg_score.sector,
            'market_cap': esg_score.market_cap,
            'last_updated': esg_score.last_updated
        }])
        
        self.esg_scores = pd.concat([self.esg_scores, new_row], ignore_index=True)
    
    def get_esg_score(self, ticker: str) -> Optional[ESGScore]:
        """Get ESG score for a specific ticker."""
        row = self.esg_scores[self.esg_scores['ticker'] == ticker]
        if row.empty:
            return None
        
        data = row.iloc[0]
        return ESGScore(
            ticker=data['ticker'],
            company_name=data['company_name'],
            environmental_score=data['environmental_score'],
            social_score=data['social_score'],
            governance_score=data['governance_score'],
            overall_esg_score=data['overall_esg_score'],
            sector=data['sector'],
            market_cap=data['market_cap'],
            last_updated=data['last_updated']
        )
    
    def get_all_esg_scores(self) -> pd.DataFrame:
        """Get all ESG scores as a DataFrame."""
        return self.esg_scores.copy()
    
    def add_market_data(self, market_data: MarketData) -> None:
        """Add or update market data for a company."""
        # Remove existing entry if it exists
        self.market_data = self.market_data[self.market_data['ticker'] != market_data.ticker]
        
        # Add new entry
        new_row = pd.DataFrame([{
            'ticker': market_data.ticker,
            'current_price': market_data.current_price,
            'volume': market_data.volume,
            'market_cap': market_data.market_cap,
            'pe_ratio': market_data.pe_ratio,
            'dividend_yield': market_data.dividend_yield,
            'beta': market_data.beta,
            'volatility': market_data.volatility,
            'last_updated': market_data.last_updated
        }])
        
        self.market_data = pd.concat([self.market_data, new_row], ignore_index=True)
    
    def get_market_data(self, ticker: str) -> Optional[MarketData]:
        """Get market data for a specific ticker."""
        row = self.market_data[self.market_data['ticker'] == ticker]
        if row.empty:
            return None
        
        data = row.iloc[0]
        return MarketData(
            ticker=data['ticker'],
            current_price=data['current_price'],
            volume=data['volume'],
            market_cap=data['market_cap'],
            pe_ratio=data['pe_ratio'],
            dividend_yield=data['dividend_yield'],
            beta=data['beta'],
            volatility=data['volatility'],
            last_updated=data['last_updated']
        )
    
    def add_recommendation(self, user_id: str, recommendation: Recommendation) -> None:
        """Add investment recommendation for a user."""
        if user_id not in self.recommendations:
            self.recommendations[user_id] = []
        
        self.recommendations[user_id].append(recommendation)
    
    def get_recommendations(self, user_id: str) -> List[Recommendation]:
        """Get all recommendations for a user."""
        return self.recommendations.get(user_id, [])
    
    def add_risk_result(self, user_id: str, risk_data: Dict[str, Any]) -> None:
        """Add risk assessment result for a user."""
        self.risk_results[user_id] = risk_data
    
    def get_risk_result(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get risk assessment result for a user."""
        return self.risk_results.get(user_id)
    
    def add_portfolio(self, user_id: str, portfolio_data: Dict[str, Any]) -> None:
        """Add portfolio data for a user."""
        self.portfolios[user_id] = portfolio_data
    
    def get_portfolio(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get portfolio data for a user."""
        return self.portfolios.get(user_id)
    
    def clear_user_data(self, user_id: str) -> None:
        """Clear all data for a specific user."""
        self.user_preferences.pop(user_id, None)
        self.recommendations.pop(user_id, None)
        self.risk_results.pop(user_id, None)
        self.portfolios.pop(user_id, None)
    
    def export_data(self) -> Dict[str, Any]:
        """Export all data for backup or debugging."""
        return {
            'user_preferences': {k: v.__dict__ for k, v in self.user_preferences.items()},
            'esg_scores': self.esg_scores.to_dict('records'),
            'market_data': self.market_data.to_dict('records'),
            'recommendations': {k: [r.__dict__ for r in v] for k, v in self.recommendations.items()},
            'risk_results': self.risk_results,
            'portfolios': self.portfolios
        }
    
    def import_data(self, data: Dict[str, Any]) -> None:
        """Import data from backup."""
        # Clear existing data
        self.user_preferences.clear()
        self.esg_scores = pd.DataFrame()
        self.market_data = pd.DataFrame()
        self.recommendations.clear()
        self.risk_results.clear()
        self.portfolios.clear()
        
        # Import user preferences
        for user_id, pref_data in data.get('user_preferences', {}).items():
            self.user_preferences[user_id] = UserPreferences(**pref_data)
        
        # Import ESG scores
        if data.get('esg_scores'):
            self.esg_scores = pd.DataFrame(data['esg_scores'])
        
        # Import market data
        if data.get('market_data'):
            self.market_data = pd.DataFrame(data['market_data'])
        
        # Import recommendations
        for user_id, recs in data.get('recommendations', {}).items():
            self.recommendations[user_id] = [Recommendation(**r) for r in recs]
        
        # Import other data
        self.risk_results = data.get('risk_results', {})
        self.portfolios = data.get('portfolios', {})


# Global instance
data_store = DataStore()
