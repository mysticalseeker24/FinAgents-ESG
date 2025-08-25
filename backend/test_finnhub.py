"""
Test script for Finnhub integration.

This script tests the Finnhub API connection and basic functionality.
"""

import json
import os
import sys
from datetime import datetime

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_finnhub_connection():
    """Test basic Finnhub connection and API calls."""
    
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "settings.json")
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print("❌ Configuration file not found")
        return False
    except json.JSONDecodeError:
        print("❌ Invalid configuration file")
        return False
    
    # Check if Finnhub API key is configured
    api_key = config.get("FINNHUB_API_KEY")
    if not api_key:
        print("❌ FINNHUB_API_KEY not found in configuration")
        return False
    
    print(f"✅ Found Finnhub API key: {api_key[:10]}...")
    
    try:
        import finnhub
        client = finnhub.Client(api_key=api_key)
        print("✅ Finnhub client initialized successfully")
    except ImportError:
        print("❌ finnhub-python package not installed")
        print("Run: pip install finnhub-python")
        return False
    except Exception as e:
        print(f"❌ Failed to initialize Finnhub client: {e}")
        return False
    
    # Test basic API calls
    try:
        # Test quote endpoint
        print("\n🔍 Testing quote endpoint...")
        quote = client.quote("AAPL")
        if quote and 'c' in quote:
            print(f"✅ AAPL quote: ${quote['c']:.2f}")
        else:
            print("❌ Failed to get AAPL quote")
            return False
        
        # Test company profile endpoint
        print("\n🔍 Testing company profile endpoint...")
        profile = client.company_profile2(symbol="AAPL")
        if profile and isinstance(profile, dict):
            company_name = profile.get('name', 'Unknown')
            sector = profile.get('finnhubIndustry', 'Unknown')
            market_cap = profile.get('marketCapitalization', 0)
            print(f"✅ AAPL profile: {company_name}")
            print(f"   Sector: {sector}")
            print(f"   Market Cap: ${market_cap/1e9:.2f}B")
        else:
            print("❌ Failed to get AAPL profile")
            return False
        
        # Test stock symbols endpoint
        print("\n🔍 Testing stock symbols endpoint...")
        symbols = client.stock_symbols('US')
        if symbols and isinstance(symbols, list):
            print(f"✅ Found {len(symbols)} US stock symbols")
            # Show first few symbols
            for i, symbol in enumerate(symbols[:5]):
                print(f"   {i+1}. {symbol.get('symbol', 'Unknown')} - {symbol.get('description', 'Unknown')}")
        else:
            print("❌ Failed to get US stock symbols")
            return False
        
        # Test candlestick data
        print("\n🔍 Testing candlestick data...")
        from datetime import timedelta
        from_timestamp = int((datetime.now() - timedelta(days=30)).timestamp())
        to_timestamp = int(datetime.now().timestamp())
        
        candles = client.stock_candles("AAPL", "D", from_timestamp, to_timestamp)
        if candles and 'c' in candles and candles['c']:
            print(f"✅ AAPL candlestick data: {len(candles['c'])} days")
            print(f"   Latest close: ${candles['c'][-1]:.2f}")
        else:
            print("❌ Failed to get AAPL candlestick data")
            return False
        
        print("\n🎉 All Finnhub API tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False


def test_esg_scoring():
    """Test ESG scoring with Finnhub data."""
    
    print("\n" + "="*50)
    print("Testing ESG Scoring with Finnhub")
    print("="*50)
    
    try:
        from esg_scoring import calculate_esg_score, calculate_detailed_esg_score
        
        # Test basic ESG score calculation
        print("\n🔍 Testing basic ESG score calculation...")
        score = calculate_esg_score("AAPL")
        print(f"✅ AAPL ESG Score: {score}")
        
        # Test detailed ESG score calculation
        print("\n🔍 Testing detailed ESG score calculation...")
        detailed = calculate_detailed_esg_score("AAPL")
        print(f"✅ AAPL Detailed ESG Score: {detailed['final_esg_score']}")
        print(f"   Company: {detailed['company_name']}")
        print(f"   Sector: {detailed['sector']}")
        print(f"   Market Cap: ${detailed['market_cap']/1e9:.2f}B")
        
        print("\n🎉 ESG scoring tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ ESG scoring test failed: {e}")
        return False


def test_market_selection():
    """Test market selection functionality."""
    
    print("\n" + "="*50)
    print("Testing Market Selection")
    print("="*50)
    
    try:
        from esg_scoring import MarketSelector, get_market_symbols
        
        # Load configuration
        config_path = os.path.join(os.path.dirname(__file__), "..", "config", "settings.json")
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Test market selector
        print("\n🔍 Testing market selector...")
        market_selector = MarketSelector(config.get("MARKET_SELECTION", {}))
        symbols = market_selector.get_filtered_tickers()
        print(f"✅ Selected {len(symbols)} symbols from market selection")
        
        # Test market-specific symbol retrieval
        print("\n🔍 Testing market-specific symbols...")
        us_symbols = get_market_symbols("US", "Technology")
        print(f"✅ US Technology symbols: {len(us_symbols)}")
        print(f"   Sample: {us_symbols[:5]}")
        
        print("\n🎉 Market selection tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Market selection test failed: {e}")
        return False


if __name__ == "__main__":
    print("🚀 Starting Finnhub Integration Tests")
    print("="*50)
    
    # Run tests
    tests = [
        ("Finnhub Connection", test_finnhub_connection),
        ("ESG Scoring", test_esg_scoring),
        ("Market Selection", test_market_selection)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name} test...")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} test passed")
            else:
                print(f"❌ {test_name} test failed")
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
    
    print("\n" + "="*50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Finnhub integration is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the configuration and dependencies.")
    
    print("="*50)
