"""
Portia AI agents for ESG investment analysis.

This module implements specialized agents using Portia AI framework
for different aspects of ESG investment analysis.
"""

import json
import os
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import pandas as pd
import numpy as np

# Import our custom modules
from storage import data_store, UserPreferences, ESGScore, MarketData, Recommendation
from esg_scoring import calculate_esg_score, calculate_detailed_esg_score, batch_calculate_esg_scores
from risk_assessment import assess_portfolio
from human_loop import HumanLoop


class ESGAdvisorAgent:
    """
    Main ESG investment advisor agent that orchestrates the entire workflow.
    
    This agent coordinates ESG scoring, risk assessment, and human approval
    to generate comprehensive investment recommendations.
    """
    
    def __init__(self, portia_api_key: str, base_url: str = "https://api.portialabs.ai"):
        """
        Initialize the ESG Advisor Agent.
        
        Args:
            portia_api_key: Portia AI API key for human approval workflows
            base_url: Portia API base URL
        """
        self.portia_api_key = portia_api_key
        self.base_url = base_url
        
        # Initialize human loop for approval workflows
        self.human_loop = HumanLoop(portia_api_key, base_url)
        
        # Load configuration
        self.config = self._load_config()
        
        # Top 50 symbols for analysis (major stocks across sectors)
        self.top_symbols = [
            # Technology
            "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "NFLX", "ADBE", "CRM",
            # Healthcare
            "JNJ", "PFE", "UNH", "ABBV", "TMO", "ABT", "LLY", "DHR", "BMY", "AMGN",
            # Financial Services
            "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "AXP", "CB", "PGR",
            # Consumer
            "PG", "KO", "PEP", "WMT", "HD", "MCD", "DIS", "NKE", "SBUX", "TGT",
            # Industrials
            "BA", "CAT", "GE", "MMM", "HON", "UPS", "FDX", "LMT", "RTX", "DE",
            # Energy & Materials
            "XOM", "CVX", "COP", "EOG", "SLB", "FCX", "NEM", "LIN", "APD", "ECL"
        ]
        
        # ESG sector preferences and exclusions
        self.esg_preferences = {
            "preferred_sectors": ["Technology", "Healthcare", "Consumer Defensive"],
            "avoided_sectors": ["Energy", "Basic Materials"],
            "min_esg_score": 60.0,
            "max_risk_volatility": 0.25
        }
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from settings.json file."""
        config_path = os.path.join(os.path.dirname(__file__), "..", "config", "settings.json")
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print("Warning: Configuration file not found, using defaults")
            return {}
        except json.JSONDecodeError:
            print("Warning: Invalid configuration file, using defaults")
            return {}
    
    def _filter_tickers_by_sector(self, tickers: List[str]) -> List[str]:
        """
        Filter tickers based on ESG sector preferences.
        
        Args:
            tickers: List of ticker symbols
            
        Returns:
            Filtered list of tickers
        """
        filtered_tickers = []
        
        for ticker in tickers:
            try:
                # Get sector information (simplified - in production you'd cache this)
                import yfinance as yf
                yf_ticker = yf.Ticker(ticker)
                info = yf_ticker.info
                sector = info.get('sector', 'Unknown')
                
                # Apply sector filtering
                if sector in self.esg_preferences["preferred_sectors"]:
                    filtered_tickers.append(ticker)
                elif sector not in self.esg_preferences["avoided_sectors"]:
                    # Include neutral sectors
                    filtered_tickers.append(ticker)
                    
            except Exception as e:
                print(f"Warning: Could not get sector info for {ticker}: {e}")
                # Include ticker if we can't determine sector
                filtered_tickers.append(ticker)
        
        return filtered_tickers
    
    def _calculate_esg_scores_for_tickers(self, tickers: List[str]) -> pd.DataFrame:
        """
        Calculate ESG scores for a list of tickers.
        
        Args:
            tickers: List of ticker symbols
            
        Returns:
            DataFrame with ticker and ESG score information
        """
        print(f"Calculating ESG scores for {len(tickers)} tickers...")
        
        esg_data = []
        
        for ticker in tickers:
            try:
                # Calculate detailed ESG score
                detailed_score = calculate_detailed_esg_score(ticker)
                
                esg_data.append({
                    'ticker': ticker,
                    'company_name': detailed_score.get('company_name', ticker),
                    'sector': detailed_score.get('sector', 'Unknown'),
                    'esg_score': detailed_score.get('final_esg_score', 0.0),
                    'market_cap': detailed_score.get('market_cap', 0),
                    'volatility': detailed_score.get('metrics', {}).get('annualized_volatility', 0.0),
                    'components': detailed_score.get('components', {}),
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                print(f"Warning: Failed to calculate ESG score for {ticker}: {e}")
                # Add placeholder data for failed calculations
                esg_data.append({
                    'ticker': ticker,
                    'company_name': ticker,
                    'sector': 'Unknown',
                    'esg_score': 0.0,
                    'market_cap': 0,
                    'volatility': 0.0,
                    'components': {},
                    'timestamp': datetime.now().isoformat()
                })
        
        # Create DataFrame and sort by ESG score
        df = pd.DataFrame(esg_data)
        df = df.sort_values('esg_score', ascending=False)
        
        return df
    
    def _select_top_n_by_esg_score(self, esg_df: pd.DataFrame, n: int) -> List[str]:
        """
        Select top N tickers by ESG score.
        
        Args:
            esg_df: DataFrame with ESG scores
            n: Number of top tickers to select
            
        Returns:
            List of top N ticker symbols
        """
        # Filter by minimum ESG score
        filtered_df = esg_df[esg_df['esg_score'] >= self.esg_preferences["min_esg_score"]]
        
        # Filter by maximum volatility
        filtered_df = filtered_df[filtered_df['volatility'] <= self.esg_preferences["max_risk_volatility"]]
        
        # Select top N
        top_n_df = filtered_df.head(n)
        
        return top_n_df['ticker'].tolist()
    
    def _assess_portfolio_risk(self, tickers: List[str]) -> Dict[str, Any]:
        """
        Assess portfolio risk for selected tickers.
        
        Args:
            tickers: List of ticker symbols
            
        Returns:
            Risk assessment results
        """
        try:
            print(f"Assessing portfolio risk for {len(tickers)} tickers...")
            risk_assessment = assess_portfolio(tickers)
            return risk_assessment
        except Exception as e:
            print(f"Warning: Portfolio risk assessment failed: {e}")
            return {
                'portfolio_metrics': {
                    'variance': 0.0,
                    'volatility': 0.0,
                    'sharpe_ratio': 0.0,
                    'number_of_stocks': len(tickers)
                },
                'error': str(e)
            }
    
    def _create_recommendation_payload(self, tickers: List[str], esg_df: pd.DataFrame, 
                                     risk_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create payload for human approval request.
        
        Args:
            tickers: List of selected tickers
            esg_df: DataFrame with ESG scores
            risk_assessment: Risk assessment results
            
        Returns:
            Formatted recommendation payload
        """
        # Get ESG scores for selected tickers
        selected_esg = esg_df[esg_df['ticker'].isin(tickers)]
        
        # Calculate portfolio allocation (equal weight for simplicity)
        allocation = [1.0 / len(tickers)] * len(tickers)
        
        # Calculate weighted average ESG score
        weighted_esg_score = np.average(selected_esg['esg_score'], weights=allocation)
        
        # Calculate portfolio metrics
        portfolio_metrics = {
            'total_stocks': len(tickers),
            'weighted_esg_score': round(weighted_esg_score, 2),
            'sector_diversity': len(selected_esg['sector'].unique()),
            'market_cap_range': {
                'min': selected_esg['market_cap'].min(),
                'max': selected_esg['market_cap'].max(),
                'total': selected_esg['market_cap'].sum()
            }
        }
        
        return {
            'user_id': 'system',  # System-generated recommendation
            'recommendation_type': 'portfolio',
            'recommendation_data': {
                'stocks': tickers,
                'allocation': allocation,
                'esg_scores': selected_esg[['ticker', 'esg_score', 'sector']].to_dict('records'),
                'portfolio_metrics': portfolio_metrics,
                'risk_assessment': risk_assessment,
                'generated_at': datetime.now().isoformat()
            },
            'requested_by': 'ESGAdvisorAgent',
            'priority': 'normal'
        }
    
    def generate_recommendations(self, n: int = 10) -> Dict[str, Any]:
        """
        Generate ESG-based investment recommendations.
        
        Args:
            n: Number of top recommendations to generate
            
        Returns:
            Dictionary with recommendations or approval status
        """
        try:
            print(f"Generating {n} ESG investment recommendations...")
            
            # Step 1: Select top 50 tickers (or use provided list)
            candidate_tickers = self.top_tickers.copy()
            print(f"Analyzing {len(candidate_tickers)} candidate tickers...")
            
            # Step 2: Filter by sector preferences
            filtered_tickers = self._filter_tickers_by_sector(candidate_tickers)
            print(f"After sector filtering: {len(filtered_tickers)} tickers")
            
            # Step 3: Calculate ESG scores for all candidates
            esg_df = self._calculate_esg_scores_for_tickers(filtered_tickers)
            
            # Step 4: Select top N by ESG score
            selected_tickers = self._select_top_n_by_esg_score(esg_df, n)
            print(f"Selected top {len(selected_tickers)} tickers by ESG score")
            
            if not selected_tickers:
                return {
                    'message': 'No suitable tickers found meeting ESG criteria',
                    'error': 'Insufficient ESG scores or all tickers filtered out'
                }
            
            # Step 5: Assess portfolio risk for selected tickers
            risk_assessment = self._assess_portfolio_risk(selected_tickers)
            
            # Step 6: Create recommendation payload
            recommendation_payload = self._create_recommendation_payload(
                selected_tickers, esg_df, risk_assessment
            )
            
            # Step 7: Request human approval
            print("Requesting human approval for recommendations...")
            approval_result = self.human_loop.request_approval(recommendation_payload)
            
            # Step 8: Return results based on approval
            if approval_result:
                # Get final ESG data for selected tickers
                final_esg_data = esg_df[esg_df['ticker'].isin(selected_tickers)]
                
                return {
                    'stocks': selected_tickers,
                    'scores': final_esg_data[['ticker', 'esg_score', 'sector']].to_dict('records'),
                    'risk': risk_assessment,
                    'portfolio_metrics': recommendation_payload['recommendation_data']['portfolio_metrics'],
                    'approval_status': 'approved',
                    'generated_at': datetime.now().isoformat()
                }
            else:
                return {
                    'message': 'Approval denied',
                    'approval_status': 'denied',
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            error_msg = f"Failed to generate recommendations: {str(e)}"
            print(f"Error: {error_msg}")
            return {
                'message': 'Failed to generate recommendations',
                'error': error_msg,
                'timestamp': datetime.now().isoformat()
            }
    
    def get_recommendation_history(self) -> List[Dict[str, Any]]:
        """
        Get history of generated recommendations.
        
        Returns:
            List of recommendation history
        """
        try:
            # Get approval requests from human loop
            approval_requests = self.human_loop.approval_requests
            
            history = []
            for request_id, request in approval_requests.items():
                if request.recommendation_type == 'portfolio':
                    history.append({
                        'request_id': request_id,
                        'user_id': request.user_id,
                        'status': request.status,
                        'requested_at': request.requested_at,
                        'approved_at': request.approved_at,
                        'approved_by': request.approved_by,
                        'rejection_reason': request.rejection_reason
                    })
            
            return history
            
        except Exception as e:
            print(f"Error getting recommendation history: {e}")
            return []
    
    def update_esg_preferences(self, new_preferences: Dict[str, Any]) -> bool:
        """
        Update ESG preferences and filtering criteria.
        
        Args:
            new_preferences: New preference settings
            
        Returns:
            True if updated successfully, False otherwise
        """
        try:
            # Validate new preferences
            if 'min_esg_score' in new_preferences:
                if not (0 <= new_preferences['min_esg_score'] <= 100):
                    raise ValueError("min_esg_score must be between 0 and 100")
            
            if 'max_risk_volatility' in new_preferences:
                if not (0 <= new_preferences['max_risk_volatility'] <= 1):
                    raise ValueError("max_risk_volatility must be between 0 and 1")
            
            # Update preferences
            self.esg_preferences.update(new_preferences)
            
            print("ESG preferences updated successfully")
            return True
            
        except Exception as e:
            print(f"Error updating ESG preferences: {e}")
            return False
    
    def get_agent_status(self) -> Dict[str, Any]:
        """
        Get current agent status and configuration.
        
        Returns:
            Dictionary with agent status information
        """
        return {
            'agent_type': 'ESGAdvisorAgent',
            'status': 'active',
            'esg_preferences': self.esg_preferences,
            'top_tickers_count': len(self.top_tickers),
            'portia_api_configured': bool(self.portia_api_key),
            'config_loaded': bool(self.config),
            'last_updated': datetime.now().isoformat()
        }


# Example usage and testing
if __name__ == "__main__":
    # Initialize agent (you would use actual API key)
    agent = ESGAdvisorAgent(portia_api_key="demo_key")
    
    print("ESG Advisor Agent initialized successfully!")
    print(f"Agent status: {json.dumps(agent.get_agent_status(), indent=2)}")
    
    # Generate recommendations
    print("\nGenerating ESG investment recommendations...")
    recommendations = agent.generate_recommendations(n=5)
    
    if 'stocks' in recommendations:
        print(f"\n✅ Recommendations approved! Top {len(recommendations['stocks'])} stocks:")
        for stock in recommendations['scores']:
            print(f"  {stock['ticker']}: ESG Score {stock['esg_score']:.1f} ({stock['sector']})")
        
        print(f"\nPortfolio Risk Metrics:")
        risk = recommendations['risk']['portfolio_metrics']
        print(f"  Volatility: {risk['volatility']:.4f}")
        print(f"  Sharpe Ratio: {risk['sharpe_ratio']:.4f}")
        print(f"  Variance: {risk['variance']:.6f}")
        
    else:
        print(f"\n❌ {recommendations['message']}")
    
    # Get recommendation history
    history = agent.get_recommendation_history()
    print(f"\nRecommendation History: {len(history)} requests")
    
    # Update preferences
    print("\nUpdating ESG preferences...")
    success = agent.update_esg_preferences({
        'min_esg_score': 70.0,
        'max_risk_volatility': 0.20
    })
    print(f"Preferences updated: {success}")
