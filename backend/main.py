"""
Main FastAPI application for the Smart ESG Investment Advisor.

This module provides the main API endpoints for ESG-based investment recommendations,
user preferences management, and portfolio analysis.
"""

import json
import os
from typing import Dict, Any, List, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from storage import data_store, UserPreferences, ESGScore, MarketData, Recommendation
from agents import ESGAdvisorAgent


# Load configuration
def load_config() -> Dict[str, Any]:
    """Load configuration from settings.json file."""
    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "settings.json")
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Configuration file not found")
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Invalid configuration file")


# Create FastAPI app
app = FastAPI(
    title="Smart ESG Investment Advisor",
    description="AI-powered ESG investment recommendations using Portia AI",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Load configuration
config = load_config()

# Initialize ESG Advisor Agent
esg_agent = ESGAdvisorAgent(
    portia_api_key=config.get("PORTIA_API_KEY", "demo_key"),
    base_url=config.get("PORTIA_BASE_URL", "https://api.portialabs.ai")
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint returning API information."""
    return {
        "message": "Smart ESG Investment Advisor API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


# ESG Advisor Agent endpoints
@app.get("/agent/status")
async def get_agent_status():
    """Get ESG Advisor Agent status and configuration."""
    try:
        return esg_agent.get_agent_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get agent status: {str(e)}")


@app.post("/agent/recommendations")
async def generate_recommendations(n: int = 10):
    """
    Generate ESG-based investment recommendations.
    
    Args:
        n: Number of top recommendations to generate (default: 10)
    
    Returns:
        ESG investment recommendations with risk assessment
    """
    try:
        if n < 1 or n > 50:
            raise HTTPException(status_code=400, detail="n must be between 1 and 50")
        
        recommendations = esg_agent.generate_recommendations(n=n)
        return recommendations
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate recommendations: {str(e)}")


@app.get("/agent/recommendations/history")
async def get_recommendation_history():
    """Get history of generated recommendations."""
    try:
        history = esg_agent.get_recommendation_history()
        return {
            "history": history,
            "count": len(history),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get recommendation history: {str(e)}")


@app.put("/agent/preferences")
async def update_agent_preferences(preferences: Dict[str, Any]):
    """
    Update ESG Advisor Agent preferences.
    
    Args:
        preferences: New preference settings
    
    Returns:
        Success confirmation
    """
    try:
        success = esg_agent.update_esg_preferences(preferences)
        if success:
            return {
                "message": "Agent preferences updated successfully",
                "preferences": esg_agent.esg_preferences,
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to update preferences")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update agent preferences: {str(e)}")


@app.get("/agent/preferences")
async def get_agent_preferences():
    """Get current ESG Advisor Agent preferences."""
    try:
        return {
            "preferences": esg_agent.esg_preferences,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get agent preferences: {str(e)}")


@app.get("/agent/test")
async def test_agent():
    """Test endpoint to verify ESG Advisor Agent functionality."""
    try:
        # Test with a small number of recommendations
        test_recommendations = esg_agent.generate_recommendations(n=3)
        
        return {
            "message": "Agent test completed successfully",
            "test_recommendations": test_recommendations,
            "agent_status": esg_agent.get_agent_status(),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "message": "Agent test failed",
            "error": str(e),
            "agent_status": esg_agent.get_agent_status(),
            "timestamp": datetime.now().isoformat()
        }


@app.post("/users/{user_id}/preferences")
async def set_user_preferences(
    user_id: str,
    preferences: UserPreferences
):
    """
    Set user investment preferences.
    
    Args:
        user_id: Unique identifier for the user
        preferences: User's investment preferences and constraints
    
    Returns:
        Confirmation message with user ID
    """
    try:
        data_store.add_user_preferences(user_id, preferences)
        return {
            "message": "User preferences updated successfully",
            "user_id": user_id,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update preferences: {str(e)}")


@app.get("/users/{user_id}/preferences")
async def get_user_preferences(user_id: str):
    """
    Get user investment preferences.
    
    Args:
        user_id: Unique identifier for the user
    
    Returns:
        User's investment preferences
    """
    preferences = data_store.get_user_preferences(user_id)
    if not preferences:
        raise HTTPException(status_code=404, detail="User preferences not found")
    
    return preferences


@app.post("/esg/scores")
async def add_esg_score(esg_score: ESGScore):
    """
    Add or update ESG score for a company.
    
    Args:
        esg_score: ESG scoring data for a company
    
    Returns:
        Confirmation message
    """
    try:
        data_store.add_esg_score(esg_score)
        return {
            "message": "ESG score added successfully",
            "ticker": esg_score.ticker,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add ESG score: {str(e)}")


@app.get("/esg/scores/{ticker}")
async def get_esg_score(ticker: str):
    """
    Get ESG score for a specific company.
    
    Args:
        ticker: Stock ticker symbol
    
    Returns:
        ESG score data for the company
    """
    esg_score = data_store.get_esg_score(ticker)
    if not esg_score:
        raise HTTPException(status_code=404, detail=f"ESG score not found for {ticker}")
    
    return esg_score


@app.get("/esg/scores")
async def get_all_esg_scores():
    """
    Get all ESG scores.
    
    Returns:
        DataFrame of all ESG scores
    """
    return data_store.get_all_esg_scores().to_dict('records')


@app.post("/market/data")
async def add_market_data(market_data: MarketData):
    """
    Add or update market data for a company.
    
    Args:
        market_data: Market data for a company
    
    Returns:
        Confirmation message
    """
    try:
        data_store.add_market_data(market_data)
        return {
            "message": "Market data added successfully",
            "ticker": market_data.ticker,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add market data: {str(e)}")


@app.get("/market/data/{ticker}")
async def get_market_data(ticker: str):
    """
    Get market data for a specific company.
    
    Args:
        ticker: Stock ticker symbol
    
    Returns:
        Market data for the company
    """
    market_data = data_store.get_market_data(ticker)
    if not market_data:
        raise HTTPException(status_code=404, detail=f"Market data not found for {ticker}")
    
    return market_data


@app.get("/recommendations/{user_id}")
async def get_recommendations(user_id: str):
    """
    Get investment recommendations for a user.
    
    Args:
        user_id: Unique identifier for the user
    
    Returns:
        List of investment recommendations
    """
    recommendations = data_store.get_recommendations(user_id)
    return {
        "user_id": user_id,
        "recommendations": recommendations,
        "count": len(recommendations),
        "timestamp": datetime.now().isoformat()
    }


@app.post("/recommendations/{user_id}")
async def add_recommendation(
    user_id: str,
    recommendation: Recommendation
):
    """
    Add investment recommendation for a user.
    
    Args:
        user_id: Unique identifier for the user
        recommendation: Investment recommendation data
    
    Returns:
        Confirmation message
    """
    try:
        data_store.add_recommendation(user_id, recommendation)
        return {
            "message": "Recommendation added successfully",
            "user_id": user_id,
            "ticker": recommendation.ticker,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add recommendation: {str(e)}")


@app.get("/risk/{user_id}")
async def get_risk_assessment(user_id: str):
    """
    Get risk assessment results for a user.
    
    Args:
        user_id: Unique identifier for the user
    
    Returns:
        Risk assessment data
    """
    risk_data = data_store.get_risk_result(user_id)
    if not risk_data:
        raise HTTPException(status_code=404, detail="Risk assessment not found")
    
    return risk_data


@app.get("/portfolio/{user_id}")
async def get_portfolio(user_id: str):
    """
    Get portfolio data for a user.
    
    Args:
        user_id: Unique identifier for the user
    
    Returns:
        Portfolio data
    """
    portfolio = data_store.get_portfolio(user_id)
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    return portfolio


@app.delete("/users/{user_id}")
async def clear_user_data(user_id: str):
    """
    Clear all data for a specific user.
    
    Args:
        user_id: Unique identifier for the user
    
    Returns:
        Confirmation message
    """
    try:
        data_store.clear_user_data(user_id)
        return {
            "message": "User data cleared successfully",
            "user_id": user_id,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear user data: {str(e)}")


@app.get("/data/export")
async def export_data():
    """
    Export all data for backup or debugging.
    
    Returns:
        Complete data export
    """
    try:
        return data_store.export_data()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export data: {str(e)}")


@app.post("/webhook/finnhub")
async def finnhub_webhook(request: Dict[str, Any]):
    """
    Handle Finnhub webhook notifications.
    
    Args:
        request: Webhook payload from Finnhub
        
    Returns:
        Confirmation message
    """
    try:
        # Verify webhook secret
        config = load_config()
        webhook_secret = config.get("FINNHUB_WEBHOOK_SECRET")
        
        # In production, you would verify the webhook signature here
        # For now, we'll just log the webhook data
        
        print(f"Received Finnhub webhook: {json.dumps(request, indent=2)}")
        
        # Process the webhook data based on type
        webhook_type = request.get("type", "unknown")
        
        if webhook_type == "trade":
            # Handle trade data
            symbol = request.get("data", {}).get("s", "")
            price = request.get("data", {}).get("p", 0)
            volume = request.get("data", {}).get("v", 0)
            timestamp = request.get("data", {}).get("t", 0)
            
            print(f"Trade: {symbol} @ ${price} (vol: {volume}) at {timestamp}")
            
        elif webhook_type == "news":
            # Handle news data
            headline = request.get("data", {}).get("headline", "")
            summary = request.get("data", {}).get("summary", "")
            
            print(f"News: {headline}")
            print(f"Summary: {summary}")
        
        return {
            "message": "Webhook received successfully",
            "type": webhook_type,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process webhook: {str(e)}")


@app.get("/finnhub/symbols/{exchange}")
async def get_finnhub_symbols(exchange: str = "US"):
    """
    Get stock symbols for a specific exchange from Finnhub.
    
    Args:
        exchange: Exchange code (US, LSE, TSE, etc.)
        
    Returns:
        List of stock symbols
    """
    try:
        config = load_config()
        api_key = config.get("FINNHUB_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="FINNHUB_API_KEY not configured")
        
        import finnhub
        client = finnhub.Client(api_key=api_key)
        
        symbols = client.stock_symbols(exchange)
        
        return {
            "exchange": exchange,
            "symbols": symbols,
            "count": len(symbols),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch symbols: {str(e)}")


@app.get("/finnhub/quote/{symbol}")
async def get_finnhub_quote(symbol: str):
    """
    Get real-time quote for a symbol from Finnhub.
    
    Args:
        symbol: Stock symbol
        
    Returns:
        Quote data
    """
    try:
        config = load_config()
        api_key = config.get("FINNHUB_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="FINNHUB_API_KEY not configured")
        
        import finnhub
        client = finnhub.Client(api_key=api_key)
        
        quote = client.quote(symbol)
        
        return {
            "symbol": symbol,
            "quote": quote,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch quote: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=config.get("API_HOST", "0.0.0.0"),
        port=config.get("API_PORT", 8000),
        reload=True
    )
