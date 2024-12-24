# Architecture Overview

## Core Components

### Frontend
- **Home.py**: Streamlit frontend for:
  - URL-based round/chain ID input
  - Matching results visualization
  - Stats and graphs display
  - Results CSV generation
  - Sharable summary export

### Core Modules
- **utils**: Data processing module
  - Indexer/DB data processing
  - Data transformation
  - Caching layer
  
- **funding_utils**: QF calculation engine
  - QF/COCM algorithms
  - Vote filtering
  - Result scaling

### Data Sources
- **GrantsDB**: PostgreSQL database for:
  - Indexer data (votes, projects, rounds)
  - Passport scores by address
  
- **Coingecko API**: Price data service
  - Matching token pricing
  - Indexer config integration

### Infrastructure  
- **Fly.io**: Application deployment platform

## Key Functions
- `load_data()`: Data loading entrypoint
- `funding_utils.calculate_matching_results()`: Matching calculation core
- `Home.display_network_graph()`: Voter network visualization

## External Dependencies
- Passport scoring system
- Grants Stack Indexer
- Grants Database 
- Token Config: `https://raw.githubusercontent.com/gitcoinco/grants-stack-indexer/main/src/config.ts`
- Coingecko API
