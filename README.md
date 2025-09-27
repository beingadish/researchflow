# ResearchFlow

> An AI-powered research agent that generates comprehensive, well-cited content on any topic using multi-agent workflows.

ReaserchFlow is an intelligent research assistant that combines the power of Google's Gemini AI with web search capabilities to produce detailed, factual content with proper citations. Unlike traditional AI responses, ResearchFlow ensures every claim is backed by verifiable sources.

## ✨ Features

- **Multi-Agent Architecture**: Two-stage workflow with specialized responder and revisor agents
- **Automatic Research**: Generates targeted search queries based on content gaps
- **Citation Management**: Automatically adds numbered references with URLs
- **Quality Control**: Built-in reflection and critique system for iterative improvement
- **Word Limit Enforcement**: Maintains concise ~250 word responses (excluding references)
- **Real-time Search**: Integrates with Tavily API for current information
- **Structured Output**: Uses Pydantic schemas for consistent, validated responses

## 🏗️ Architecture

ResearchFlow uses a LangGraph-based multi-agent system:

```
User Query → First Responder → Tool Execution → Revisor → Final Answer
               ↓                    ↓              ↓
          Initial Answer      Web Search     Refined Answer
          + Search Queries    + Results      + Citations
          + Self Critique                    + References
```

### Core Components

- **First Responder**: Generates initial answers with self-reflection and research queries
- **Tool Executor**: Performs web searches using Tavily API
- **Revisor**: Refines answers using search results and adds proper citations
- **Schema Validator**: Ensures structured output using Pydantic models

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Google AI API key (Gemini)
- Tavily API key

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/researchflow.git
cd researchflow
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys:
# GOOGLE_API_KEY=your_gemini_api_key
# TAVILY_API_KEY=your_tavily_api_key
```

### Usage

```python
from main import app

# Ask any research question
query = "How can small businesses leverage AI to grow?"
response = app.invoke(query)
answer = response[-1].tool_calls[0]["args"]["answer"]
print(answer)
```

Or run the interactive version:
```bash
python main.py
```

## 📝 Example Output

**Query**: "Write about the growth of Indian Football Team"

**Response**:
The Indian football team's "golden era" (1950s-60s) included an Olympic semi-final in 1956 and two Asian Games gold medals in 1951 and 1962 . After a prolonged decline, a resurgence began in the early 21st century, largely led by captain Sunil Chhetri.

India's FIFA ranking improved, breaking into the top 100 in 2017, reaching 96th . The team has consistently won the SAFF Championship and qualified for the AFC Asian Cup in 2011, 2019, and 2023 ...

**References:**
-  [https://www.olympics.com/en/news/history-of-indian-football](https://www.olympics.com/en/news/history-of-indian-football)
-  [https://www.olympics.com/en/news/india-football-team-rankings-world-fifa-best-worst-position-points-table](https://www.olympics.com/en/news/india-football-team-rankings-world-fifa-best-worst-position-points-table)

## 🛠️ Configuration

### Environment Variables

```env
# Required
GOOGLE_API_KEY=your_gemini_api_key
TAVILY_API_KEY=your_tavily_search_key

# Optional
MAX_TOOL_CALLS=3
SEARCH_RESULTS_LIMIT=5
TARGET_WORD_COUNT=250
```

### Customization

- **Model Selection**: Change the Gemini model in `chains.py`
- **Search Provider**: Replace Tavily with other search tools in `execute_tools.py`
- **Word Limits**: Modify target length in prompt templates
- **Tool Call Limits**: Adjust `MAX_TOOL_CALLS` in `main.py`

## 📁 Project Structure

```
researchflow/
├── main.py              # Main application entry point
├── chains.py            # LangChain prompt templates and chains
├── execute_tools.py     # Search tool execution logic
├── schema.py            # Pydantic data models
├── requirements.txt     # Python dependencies
├── .env.example        # Environment variables template
└── README.md           # This file
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
pip install -e .
pip install -r requirements-dev.txt
pre-commit install
```

## 📊 Performance

- **Response Time**: ~10-15 seconds per query
- **Accuracy**: High-quality responses with verifiable sources
- **Cost**: Efficient token usage with targeted search queries
- **Scalability**: Stateless design supports concurrent requests

## 🗺️ Roadmap

- [ ] Support for multiple output formats (JSON, CSV, PDF)
- [ ] Integration with academic databases (arXiv, PubMed)
- [ ] Multilingual support
- [ ] Advanced citation formats (APA, MLA, Chicago)
- [ ] Web interface and API endpoints
- [ ] Batch processing capabilities
- [ ] Custom knowledge base integration

## 🙏 Acknowledgments

- [LangChain](https://langchain.com/) for the agent framework
- [Google AI](https://ai.google.dev/) for Gemini model access
- [Tavily](https://tavily.com/) for web search capabilities
- [Pydantic](https://pydantic.dev/) for data validation

***

**ResearchFlow** - *Where AI meets rigorous research*