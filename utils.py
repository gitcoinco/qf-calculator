import streamlit as st
import pandas as pd
import numpy as np
import requests
import re
from datetime import datetime, timezone
import psycopg2 as pg
import json
import time

ttl_short = 900 # 15 minutes
ttl_long = 36000 # 10 hours

def run_query(query, params=None, database="grants"):
    """Run a parameterized query on the specified database and return results as a DataFrame."""
    try:
        conn = pg.connect(host=st.secrets[database]["host"], 
                            port=st.secrets[database]["port"], 
                            dbname=st.secrets[database]["dbname"], 
                            user=st.secrets[database]["user"], 
                            password=st.secrets[database]["password"])
        cur = conn.cursor()
        if params is None:
            cur.execute(query)
        else:
            cur.execute(query, params)
        col_names = [desc[0] for desc in cur.description]
        results = pd.DataFrame(cur.fetchall(), columns=col_names)
    except pg.Error as e:
        st.warning(f"ERROR: Could not execute the query. {e}")
    finally:
        cur.close()
        conn.close()
    return results

def load_data_from_url(url):
    """Load JSON data from a given URL and return as a list of dictionaries."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an error for bad responses
        lines = (line.decode('utf-8') for line in response.iter_lines())
        data = [json.loads(line) for line in lines if line]  # Ignore blank lines
        return data
    except requests.RequestException as e:
        st.warning(f"Failed to fetch data from {url}. Error: {e}")
    except json.JSONDecodeError as e:
        st.warning(f"Failed to parse JSON data from {url}. Error: {e}")
        return []

@st.cache_resource(ttl=0)
def get_round_summary():
    """Fetch and return a summary of all rounds from the indexer."""
    sql_query_file = 'queries/get_rounds_summary_from_indexer.sql'
    with open(sql_query_file, 'r') as file:
        query = file.read()
    results = run_query(query)
    return results

@st.cache_resource(ttl=ttl_short)
def get_round_votes(round_id, chain_id):
    """Fetch and return votes for a specific round and chain."""
    sql_query_file = 'queries/get_votes_by_round_id_from_indexer.sql'
    with open(sql_query_file, 'r') as file:
        query = file.read()
    params = {
        'round_id': round_id,
        'chain_id': chain_id
    }
    results = run_query(query, params)
    return results

@st.cache_resource(ttl=ttl_short)
def get_projects_in_round(round_id, chain_id):
    """Fetch and return projects for a specific round and chain."""
    sql_query_file = 'queries/get_projects_summary_from_indexer.sql'
    with open(sql_query_file, 'r') as file:
        query = file.read()
    params = {
        'round_id': round_id,
        'chain_id': chain_id
    }
    results = run_query(query, params)
    return results

@st.cache_resource(ttl=ttl_long) 
def load_passport_model_scores(addresses):
    """Load and process passport model scores for given addresses."""
    addresses = tuple(addresses)
    sql_query_file = 'queries/get_passport_aggregate_model_scores.sql'
    with open(sql_query_file, 'r') as file:
        query = file.read()
    params = {
        'addresses': addresses
    }
    results = run_query(query, params)
    # Load the parquet file
    df = pd.read_parquet('data/gg21_donors_scored.parquet')
    df = df[['Address', 'aggregate_score']]
    df.columns = ['address', 'rawScore']

    address_set = set(addresses)
    missing_addresses = df[df['address'].isin(address_set) & ~df['address'].isin(results['address'])]
    results = pd.concat([results, missing_addresses], ignore_index=True)
    

    return results

@st.cache_resource(ttl=ttl_long)
def load_avax_scores(addresses):
    """Load and process Avalanche scores for given addresses."""
    url = 'https://public.scorer.gitcoin.co/passport_scores/6608/registry_score.jsonl'
    scores = load_data_from_url(url)
    scores = pd.DataFrame(scores)
    scores = scores.join(pd.json_normalize(scores['evidence'])).drop('evidence', axis=1)
    scores = scores.join(pd.json_normalize(scores['passport'])).drop('passport', axis=1) 
    scores['CivicUniquenessPass'] = scores['stamp_scores'].apply(lambda x: x.get('CivicUniquenessPass', 0))
    scores['HolonymGovIdProvider'] = scores['stamp_scores'].apply(lambda x: x.get('HolonymGovIdProvider', 0))
    scores = scores[scores['address'].isin(addresses)]
    scores = scores.sort_values('last_score_timestamp', ascending=False).drop_duplicates('address')
    scores['score'] = scores['score'].astype(float)
    scores['rawScore'] = scores['rawScore'].astype(float)
    return scores

@st.cache_resource(ttl=ttl_long)
def load_stamp_scores(addresses):
    """Load and process passport stamp scores for given addresses."""
    addresses = tuple(addresses)
    sql_query_file = 'queries/get_passport_stamps.sql'
    with open(sql_query_file, 'r') as file:
        query = file.read()
    params = {
        'addresses': addresses
    }
    results = run_query(query, params)  
    return results

def parse_config_file(file_content):
    """Parse the config file content and extract token information."""
    data = []
    chain_pattern = re.compile(r'{\s*id:\s*(\d+),\s*name:\s*"([^"]+)",.*?tokens:\s*\[(.*?)\].*?}', re.DOTALL)
    token_pattern = re.compile(r'code:\s*"(?P<code>[^"]+)".*?address:\s*"(?P<address>[^"]+)".*?decimals:\s*(?P<decimals>\d+).*?priceSource:\s*{\s*chainId:\s*(?P<price_source_chain_id>\d+).*?address:\s*"(?P<price_source_address>[^"]+)"', re.DOTALL)
    chain_matches = chain_pattern.findall(file_content)

    for chain_match in chain_matches:
        chain_id = int(chain_match[0])
        chain_name = chain_match[1]
        token_data = chain_match[2]

        token_matches = token_pattern.finditer(token_data)

        for token_match in token_matches:
            token_code = token_match.group('code')
            token_address = token_match.group('address')
            token_decimals = int(token_match.group('decimals'))
            price_source_chain_id = int(token_match.group('price_source_chain_id'))
            price_source_address = token_match.group('price_source_address')

            data.append([
                chain_id,
                chain_name,
                token_code,
                token_address,
                token_decimals,
                price_source_chain_id,
                price_source_address
            ])

    if data:
        columns = [
            'chain_id',
            'chain_name',
            'token_code',
            'token_address',
            'token_decimals',
            'price_source_chain_id',
            'price_source_address'
        ]
        df = pd.DataFrame(data, columns=columns)
        df['token_address'] = df['token_address'].str.lower()
        df['price_source_address'] = df['price_source_address'].str.lower()
        return df
    else:
        print("No token data found in the file.")
        return None
    
@st.cache_resource(ttl=ttl_long)
def fetch_tokens_config():
    """Fetch and parse the token configuration from the GitHub repository."""
    url = 'https://raw.githubusercontent.com/gitcoinco/grants-stack-indexer/main/src/config.ts'
    
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad responses
    except requests.RequestException as e:
        print(f"Failed to fetch data from {url}. Error: {e}")
        return None

    df = parse_config_file(response.text)
    return df

@st.cache_resource(ttl=ttl_long)
def fetch_latest_price(chain_id, token_address, coingecko_api_key=st.secrets['coingecko']['COINGECKO_API_KEY'], coingecko_api_url="https://api.coingecko.com/api/v3"):
    """Fetch the latest price for a given token on a specific chain."""
    # https://github.com/gitcoinco/grants-stack-indexer/blob/main/src/prices/coinGecko.ts
    platforms = {
        1: "ethereum",
        250: "fantom",
        10: "optimistic-ethereum",
        42161: "arbitrum-one",
        43114: "avalanche",
        713715: "sei-devnet",
        1329: "sei-mainnet",
        42220: "celo",
        1088: "metisAndromeda",
        42: "lukso-mainnet"
    }

    native_tokens = {
        1: "ethereum",
        250: "fantom",
        10: "ethereum",
        42161: "ethereum",
        43114: "avalanche-2",
        713715: "sei-network",
        1329: "sei-network",
        42220: "celo-mainnet",
        1088: "metis",
        42: "lukso-token"

    }

    if chain_id not in platforms:
        raise ValueError(f"Prices for chain ID {chain_id} are not supported.")

    is_native_token = token_address == "0x0000000000000000000000000000000000000000"
    platform = platforms[chain_id]

    if is_native_token:
        path = f"/simple/price?ids={native_tokens[chain_id]}&vs_currencies=usd"
        key = native_tokens[chain_id]
    else:
        path = f"/simple/token_price/{platform}?contract_addresses={token_address}&vs_currencies=usd"
        key = token_address

    headers = {
        "accept": "application/json",
        "x-cg-demo-api-key": coingecko_api_key
    }

    max_retries = 4
    retry_delay = 4  # seconds

    for retry_count in range(max_retries):
        response = requests.get(f"{coingecko_api_url}{path}", headers=headers)

        if response.status_code == 429:
            if retry_count == max_retries - 1:
                raise ValueError("CoinGecko API rate limit exceeded, are you using an API key?")
            time.sleep(retry_delay)
        else:
            break

    response_data = response.json()

    if "error" in response_data:
        raise ValueError(f"Error from CoinGecko API: {response_data}")
        
    if key not in response_data:
        raise ValueError(f"Token {'native' if is_native_token else 'address'} '{key}' not found in the response data.")
        
    return response_data[key]["usd"]