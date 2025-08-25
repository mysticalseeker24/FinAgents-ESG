# Smart ESG Investment Advisor

A comprehensive investment advisory system that leverages ESG (Environmental, Social, and Governance) criteria to provide intelligent investment recommendations using Portia AI's multi-agent framework.

## Project Overview

The Smart ESG Investment Advisor combines FastAPI backend services with a Streamlit frontend to deliver ESG-focused investment insights. The system utilizes multiple specialized agents powered by Portia AI to analyze market data, assess ESG metrics, and generate personalized investment strategies.

### Architecture

- **Backend**: FastAPI with in-memory data storage
- **Frontend**: Streamlit interactive interface
- **AI Framework**: Portia SDK for multi-agent orchestration
- **Data Sources**: yfinance for market data, ESG databases
- **LLM Provider**: Blackbox AI (o4-mini-high model)
- **Storage**: In-memory data structures (no database required)

*[Architecture diagram will be added here]*

## Features

- ESG scoring and analysis
- Risk assessment and portfolio optimization
- Real-time market data integration
- Multi-agent decision making
- Interactive data visualization
- Personalized investment recommendations
- In-memory data storage for rapid development

## Prerequisites

- Python 3.10 or higher
- Virtual environment (recommended)
- Portia API key
- Blackbox AI API key
- Finnhub API key (free tier available)

## Setup Instructions

### 1. Clone and Navigate
```bash
cd PORTIA-AGENTHACKS
```

### 2. Create Virtual Environment
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Unix/MacOS
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Environment Configuration
Create a `.env` file in the root directory:
```bash
PORTIA_API_KEY=your_portia_api_key_here
BLACKBOX_API_KEY=your_blackbox_api_key_here
FINNHUB_API_KEY=your_finnhub_api_key_here
```

### 5. Configuration Setup
Update `config/settings.json` with your API keys:
```json
{
  "PORTIA_API_KEY": "your_actual_portia_api_key",
  "BLACKBOX_API_KEY": "your_actual_blackbox_api_key",
  "FINNHUB_API_KEY": "your_actual_finnhub_api_key"
}
```

## Running the Application

### Backend (FastAPI)
```bash
cd backend
python main.py
```

### Frontend (Streamlit)
```bash
cd frontend
streamlit run app.py
```

The application will be available at:
- Backend API: http://localhost:8010
- Frontend: http://localhost:8501
- API Documentation: http://localhost:8010/docs

## Project Structure

```
PORTIA-AGENTHACKS/
├── backend/                 # FastAPI backend application
│   ├── main.py             # FastAPI app entry point and routes
│   ├── storage.py          # In-memory data storage
│   ├── agents.py           # Portia AI agents (to be implemented)
│   ├── esg_scoring.py      # ESG scoring logic (to be implemented)
│   ├── risk_assessment.py  # Risk assessment (to be implemented)
│   └── human_loop.py       # Human-in-the-loop features (to be implemented)
├── frontend/               # Streamlit frontend application
│   └── app.py              # Main Streamlit application (to be implemented)
├── config/                 # Configuration files
│   └── settings.json       # Application settings and API keys
├── cursorrules.yaml        # AI generation rules
├── requirements.txt         # Python dependencies
└── README.md               # Project documentation
```

## Technology Stack

- **[FastAPI](https://fastapi.tiangolo.com/)** - Modern, fast web framework for building APIs
- **[Portia SDK](https://github.com/portiaAI/portia-sdk-python)** - Multi-agent AI framework
- **[Streamlit](https://streamlit.io/)** - Rapid web app development framework
- **[Finnhub](https://finnhub.io/)** - Professional market data API with real-time feeds
- **[Plotly](https://plotly.com/python/)** - Interactive plotting library
- **[Blackbox AI](https://www.blackbox.ai/dashboard/docs)** - LLM provider (o4-mini-high model)

## Key Benefits

- **No Database Setup** - Uses in-memory storage for rapid development
- **Hackathon Friendly** - Simple deployment and demo setup
- **Portable** - Can run anywhere without external dependencies
- **Fast Development** - Focus on core AI features rather than infrastructure
- **Professional Data** - Uses Finnhub API for reliable, real-time market data
- **Rate Limiting** - Built-in rate limiting prevents API quota issues
- **Webhook Support** - Real-time data updates via webhooks

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the Apache License 2.0.

## Support

For issues and questions:
- Check the [Portia SDK documentation](https://github.com/portiaAI/portia-sdk-python)
- Review [Blackbox AI documentation](https://www.blackbox.ai/dashboard/docs)
- Review FastAPI and Streamlit documentation
- Open an issue in this repository
