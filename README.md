# 🧮 Quadratic Funding Calculator

## 📋 Overview
This project is a Streamlit-based web application that calculates and visualizes matching results for Quadratic Funding (QF) rounds. It compares standard Quadratic Funding with Connection-Oriented Cluster Matching (COCM) to provide insights into fund distribution.

## ✨ Features
- Load and process round data from various blockchain networks
- Apply Sybil defense mechanisms using Passport scores
- Calculate and compare matching results using QF and COCM algorithms
- Visualize crowdfunding statistics and donation distributions
- Generate downloadable matching distributions and round summaries
- Filter out specific wallets or projects from calculations

## 🚀 Installation

1. Clone the repository:
   ```
   git clone [repository-url]
   cd [repository-name]
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Prep

1. Install and setup [Indexer Data](https://github.com/gitcoinco/indexer-data)

2. Setup Hasura.
   1. Run Docker compose file: `cd hasura && docker compose up -d`
   2. Visit `http://localhost:8080/console`
      1. Click on `Data`
      2. Click on `Data Manager`
      3. Select `Postgres`
      4. Click `Connect Existing Database`
      5. For Database name, use whatever name you want, example `indexer`
      6. under `Connect Database via:`, select `Database URL` and enter your indexer db URL
      7. Expand `GraphQL Customization` and change `Naming Convention` from `hasura-default` to `graphql-default`.
      8. Click `Connect Database`
      9. Under `Databases` on the sidebar, select your newly added DB, select `public`
         1. Under `Untracked tables or views`, click `Track All` and confirm.
         2. Under `Untracked foreign-key relationships`, click `Track All` and confirm
         3. Under `Untracked custom functions`, click `Track`, then `Add As Root Field`
      10. Click on `API` at the top nav, and confirm your queries work.

3. Obtain Passport & Coingecko API key

4. Setup streamlit secrets. You can create a `.streamlit/secrets.toml` file with below content:

```toml
[config]
BASE_URL = "http://localhost:8501"
GRAPHQL_URL = "http://hasura_graphql_endpoint"

[grants]
host = "indexer_db_host"
port = "indexer_db_port"
dbname = "indexer_db_dbName"
user = "indexer_db_username"
password = "indexer_db_password"

[passport]
key = "passport_key"

[coingecko]
COINGECKO_API_KEY = "coingecko_key"
```

## 🖥️ Usage

1. Run the Streamlit app:
   ```
   streamlit run Home.py
   ```

2. Access the app through your web browser, typically at `http://localhost:8501`

3. Provide the `round_id` and `chain_id` as URL parameters:
   ```
   http://localhost:8501/?round_id=[ROUND_ID]&chain_id=[CHAIN_ID]
   ```

## ⚙️ Configuration

- The app uses environment variables for database connections and API keys. Ensure these are set up in a `secrets.toml` file or your environment.

## 📁 Files Description

- `Home.py`: Main Streamlit application file
- `fundingutils.py`: Contains functions for QF calculations
- `utils.py`: Utility functions for data loading and processing
- `requirements.txt`: List of Python package dependencies
- `queries/`: SQL query files for data retrieval

## 📦 Dependencies

- streamlit
- pandas
- numpy
- plotly
- psycopg2-binary

## 🚀 Deployment with Fly.io

This project is configured for deployment on Fly.io. Here are the steps to deploy:

1. Install the Fly CLI: Follow the instructions at [https://fly.io/docs/hands-on/install-flyctl/](https://fly.io/docs/hands-on/install-flyctl/)

2. Login to Fly:
   ```
   fly auth login
   ```

3. Navigate to your project directory and initialize the Fly app:
   ```
   fly launch
   ```

4. Deploy the app:
   ```
   fly deploy
   ```

5. Once deployed, you can access your app at `https://qf-calculator.fly.dev`

Remember to set up your environment variables and secrets in the Fly.io dashboard or using the Fly CLI before deployment.

For more detailed information on deploying Streamlit apps on Fly.io, refer to their documentation: [https://fly.io/docs/app-guides/streamlit/](https://fly.io/docs/app-guides/streamlit/)