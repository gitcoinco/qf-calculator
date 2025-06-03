import streamlit as st
import pandas as pd
import numpy as np
import requests
import re
from datetime import datetime, timezone
import psycopg2 as pg
import json
import time
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed

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

def load_data_from_url(url, headers=None, show_warnings=True):
    """Load JSON data from a given URL and return as a list of dictionaries."""
    try:
        response = requests.get(url, headers=headers, stream=True)
        response.raise_for_status()  # Raise an error for bad responses
        lines = (line.decode('utf-8') for line in response.iter_lines())
        data = [json.loads(line) for line in lines if line]  # Ignore blank lines
        return data
    except requests.RequestException as e:
        if show_warnings:
            st.warning(f"Failed to fetch data from {url}. Error: {e}")
        raise  # Re-raise so the calling function can handle it
    except json.JSONDecodeError as e:
        if show_warnings:
            st.warning(f"Failed to parse JSON data from {url}. Error: {e}")
        return []

def _fetch_single_passport_model_score(address: str, api_base_url: str, headers: dict) -> dict:
    """Helper function to fetch a single passport model score."""
    try:
        model_url = f"{api_base_url}/v2/models/score/{address}"
        model_response = load_data_from_url(model_url, headers, show_warnings=False)

        if model_response:
            data = model_response[0]  # load_data_from_url returns a list

            print("DATA: ", data)
            
            return {
                'address': address.lower(),
                'rawScore': float(data.get('details', {}).get('models', {}).get('aggregate', {}).get('score', 0)),
                'updated_at': data.get('last_score_timestamp', None),
                'error': None
            }
        else:
            # Handle API failure case
            return {
                'address': address.lower(),
                'rawScore': 0,
                'updated_at': None,
                'error': 'API returned no data'
            }
            
    except Exception as e:
        # Don't use st.warning() in thread - return error info instead
        return {
            'address': address.lower(),
            'rawScore': 0,
            'updated_at': None,
            'error': str(e)
        }

# @st.cache_resource(ttl=ttl_long)
def load_passport_model_scores(addresses: List[str]) -> pd.DataFrame:
    """Load passport model scores from Gitcoin Passport API v2 with parallel requests."""
    
    # API v2 configuration
    API_BASE_URL = "https://api.passport.xyz"
    headers = {
        'X-API-KEY': st.secrets['passport']['key'],
        'Content-Type': 'application/json'
    }
    
    processed_data = []
    errors = []
    
    # Use ThreadPoolExecutor for parallel API calls
    with ThreadPoolExecutor(max_workers=5) as executor:
        # Submit all requests
        future_to_address = {
            executor.submit(_fetch_single_passport_model_score, address, API_BASE_URL, headers): address 
            for address in addresses
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_address):
            result = future.result()
            
            # Collect errors to display later in main thread
            if result.get('error'):
                errors.append(f"Error fetching model score for {result['address']}: {result['error']}")
            
            # Remove error field before adding to processed_data
            result_clean = {k: v for k, v in result.items() if k != 'error'}
            processed_data.append(result_clean)
    
    # Display any errors that occurred (now in main thread with Streamlit context)
    for error in errors:
        st.warning(error)
    
    # Create DataFrame
    results = pd.DataFrame(processed_data)
    
    # Keep the historical data fallback logic
    df = pd.read_parquet('data/gg21_donors_scored.parquet')
    df = df[['Address', 'aggregate_score']]
    df.columns = ['address', 'rawScore']

    df_22 = pd.read_csv('data/gg22_donors_scored.csv')
    df_22 = df_22[['Address', 'aggregate_score']]
    df_22.columns = ['address', 'rawScore']
    df_22['rawScore'] = df_22['rawScore'].fillna(0)

    df_23 = pd.read_csv('data/gg23_donors_scored.csv')
    df_23 = df_23[['Address', 'aggregate_score']]
    df_23.columns = ['address', 'rawScore']
    df_23['rawScore'] = df_23['rawScore'].fillna(0)

    df = pd.concat([df, df_22, df_23], ignore_index=True)
    df['address'] = df['address'].str.lower()
    df = df.drop_duplicates(subset='address', keep='last')

    address_set = set([addr.lower() for addr in addresses])
    missing_addresses = df[df['address'].isin(address_set) & ~df['address'].isin(results['address'])]
    results = pd.concat([results, missing_addresses], ignore_index=True)
    
    return results

@st.cache_resource(ttl=ttl_long)
def load_stamp_scores(addresses: List[str]) -> pd.DataFrame:
    """Load and process passport stamp scores from Gitcoin Passport API v2."""
    
    # API v2 configuration
    API_BASE_URL = "https://api.passport.xyz"
    headers = {
        'X-API-KEY': st.secrets['passport']['key'],
        'Content-Type': 'application/json'
    }
    
    processed_data = []
    scorer_id = st.secrets['config']['SCORER_ID']
    
    for address in addresses:
        try:
            # V2 API: Single endpoint for both stamps and score data
            score_url = f"{API_BASE_URL}/v2/stamps/{scorer_id}/score/{address}"
            score_response = load_data_from_url(score_url, headers)
            
            if score_response:
                data = score_response[0]  # load_data_from_url returns a list
                
                # Extract stamp data from stamp_scores
                stamps = []
                if 'stamp_scores' in data:
                    for provider, score in data['stamp_scores'].items():
                        stamps.append({
                            'provider': provider,
                            'score': float(score) if score else 0,
                            'verified': float(score) > 0 if score else False
                        })
                
                # If you need detailed stamp metadata, fetch from stamps endpoint
                stamps_url = f"{API_BASE_URL}/v2/stamps/{address}"
                stamps_response = load_data_from_url(stamps_url, headers)
                
                if stamps_response and stamps_response[0].get('items'):
                    # Enhance stamps data with detailed metadata
                    detailed_stamps = []
                    for item in stamps_response[0]['items']:
                        credential = item.get('credential', {})
                        credential_subject = credential.get('credentialSubject', {})
                        
                        stamp_info = {
                            'provider': credential_subject.get('provider'),
                            'hash': credential_subject.get('hash'),
                            'issuanceDate': credential.get('issuanceDate'),
                            'expirationDate': credential.get('expirationDate'),
                            'score': data.get('stamp_scores', {}).get(credential_subject.get('provider'), 0)
                        }
                        detailed_stamps.append(stamp_info)
                    stamps = detailed_stamps
                
                processed_data.append({
                    'address': address.lower(),
                    'rawScore': float(data.get('score', 0)),
                    'scoreTimestamp': data.get('last_score_timestamp'),
                    'updatedAt': data.get('last_score_timestamp'),  # v2 doesn't have separate updated_at
                    'stamps': stamps
                })
            else:
                # Handle API failure case
                processed_data.append({
                    'address': address.lower(),
                    'rawScore': 0,
                    'scoreTimestamp': None,
                    'updatedAt': None,
                    'stamps': []
                })
                
        except Exception as e:
            st.warning(f"Error fetching passport data for {address}: {e}")
            processed_data.append({
                'address': address.lower(),
                'rawScore': 0,
                'scoreTimestamp': None,
                'updatedAt': None,
                'stamps': []
            })
    
    return pd.DataFrame(processed_data)

@st.cache_resource(ttl=ttl_long)
def load_avax_scores(addresses):

    """Load and process Avalanche scores for given addresses."""

    scores = pd.concat([pd.DataFrame(load_data_from_url(f'https://api.passport.xyz/v2/stamps/335/score/{a}', headers = {'X-API-KEY': st.secrets['passport']['key']})) for a in addresses])

    #scores = scores.join(pd.json_normalize(scores['evidence'])).drop('evidence', axis=1)
    #scores = scores.join(pd.json_normalize(scores['passport'])).drop('passport', axis=1) 
    #scores['CivicUniquenessPass'] = scores['stamp_scores'].apply(lambda x: x.get('CivicUniquenessPass', 0))
    #scores['HolonymGovIdProvider'] = scores['stamp_scores'].apply(lambda x: x.get('HolonymGovIdProvider', 0))

    scores = scores[scores['address'].isin(addresses)]
    scores = scores.sort_values('last_score_timestamp', ascending=False).drop_duplicates('address')
    scores['score'] = scores['score'].astype(float)
    scores['rawScore'] = scores['score'].astype(float)
    return scores

@st.cache_resource(ttl=ttl_long)
def load_stamp_scores_unified(addresses: List[str]) -> pd.DataFrame:
    """Simplified version using v2 unified endpoint for stamps and scores."""
    
    headers = {
        'X-API-KEY': st.secrets['passport']['key'],
        'Content-Type': 'application/json'
    }
    
    processed_data = []
    scorer_id = st.secrets['config']['SCORER_ID']
    
    for address in addresses:
        # Single API call gets both score and stamp data
        url = f"https://api.passport.xyz/v2/stamps/{scorer_id}/score/{address}"
        response = load_data_from_url(url, headers)
        
        if response:
            data = response[0]
            
            # Convert stamp_scores to the expected stamps format
            stamps = [
                {
                    'provider': provider,
                    'score': float(score) if score else 0,
                    'verified': float(score) > 0 if score else False
                }
                for provider, score in data.get('stamp_scores', {}).items()
            ]
            
            processed_data.append({
                'address': address.lower(),
                'rawScore': float(data.get('score', 0)),
                'scoreTimestamp': data.get('last_score_timestamp'),
                'updatedAt': data.get('last_score_timestamp'),
                'stamps': stamps
            })
        else:
            processed_data.append({
                'address': address.lower(),
                'rawScore': 0,
                'scoreTimestamp': None,
                'updatedAt': None,
                'stamps': []
            })
    
    return pd.DataFrame(processed_data)

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
        st.write(f"Failed to fetch data from {url}. Error: {e}")
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
