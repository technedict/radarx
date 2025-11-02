# Getting Started with RadarX

## Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager
- Virtual environment (recommended)

### Install from Source

```bash
# Clone the repository
git clone https://github.com/technedict/radarx.git
cd radarx

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## Configuration

Create a `.env` file in the project root:

```bash
# API Settings
API_HOST=0.0.0.0
API_PORT=8000

# Database
DATABASE_URL=sqlite:///./radarx.db
REDIS_URL=redis://localhost:6379/0

# API Keys (optional for demo)
DEXSCREENER_API_KEY=your_key_here
ETHERSCAN_API_KEY=your_key_here
TWITTER_BEARER_TOKEN=your_token_here

# Model Settings
MODEL_VERSION=v1.2.3
FEATURE_VERSION=v2.0.1
ENABLE_ONLINE_LEARNING=true
```

## Running the API Server

Start the API server:

```bash
# Using Python module
python -m radarx.api.server

# Or use the CLI command
radarx-server
```

The API will be available at `http://localhost:8000`. Visit `http://localhost:8000/docs` for interactive API documentation.

## First API Call

Try scoring a token:

```bash
curl "http://localhost:8000/score/token?address=0x1234567890abcdef1234567890abcdef12345678&chain=ethereum&horizons=24h,7d"
```

## Next Steps

- Read the [API Examples](api-examples.md) for more usage patterns
- Check out the [User Guide](user-guide.md) for detailed features
- See [API Reference](api-reference.md) for complete endpoint documentation
