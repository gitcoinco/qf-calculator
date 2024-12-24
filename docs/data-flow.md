# Data Flow


## Critical Paths

### 1. Round Initialization (`Home.py`)
- Load config from URL params
- Validate round exists in Grants DB  
- Set up chain/token config from Indexer Config File

### 2. Data Loading (`utils`)
- Fetch data from Grants DB using SQL queries:
  - Votes (4 hour refresh)
  - Round data settings & QF parameters (4 hour refresh) 
  - Passport scores (daily refresh)
- Transform data
- Visualize data

### 3. Matching Calculations (`funding_utils`) 
- Filter ineligible votes:
  - Self-votes
  - Below minimum amount
  - Insufficient passport score
- Run QF and/or COCM algorithms
- Apply caps and scaling
- Handle overflows

### 4. Export Generation (`Home.py`)
- Format results as:
  - Matching Distribution CSV for onchain upload
  - Sharable Summary CSV for public viewing
