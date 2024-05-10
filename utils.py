import streamlit as st
import pandas as pd
import numpy as np
import requests
import re
from datetime import datetime, timezone
import psycopg2 as pg
import json
from dune_client.types import QueryParameter
from dune_client.client import DuneClient
import time


@st.cache_resource(ttl=36000)
def run_query(query):
    """Run query and return results"""
    try:
        conn = pg.connect(host=st.secrets["indexer"]["host"], 
                           port=st.secrets["indexer"]["port"], 
                           dbname=st.secrets["indexer"]["dbname"], 
                           user=st.secrets["indexer"]["user"], 
                           password=st.secrets["indexer"]["password"])
        cur = conn.cursor()
        cur.execute(query)
        col_names = [desc[0] for desc in cur.description]
        results = pd.DataFrame(cur.fetchall(), columns=col_names)
    except pg.Error as e:
        st.warning(f"ERROR: Could not execute the query. {e}")
    finally:
        conn.close()
    return results

@st.cache_resource(ttl=36000)
def run_query_with_params(query, params):
    conn = pg.connect(host=st.secrets["indexer"]["host"], 
                           port=st.secrets["indexer"]["port"], 
                           dbname=st.secrets["indexer"]["dbname"], 
                           user=st.secrets["indexer"]["user"], 
                           password=st.secrets["indexer"]["password"])
    cur = conn.cursor()
    cur.execute(query, params)
    col_names = [desc[0] for desc in cur.description]
    results = pd.DataFrame(cur.fetchall(), columns=col_names)
    cur.close()
    conn.close()
    return results

def run_grants_query_with_params(query, params):
    conn = pg.connect(     host=st.secrets["grants"]["host"], 
                           port=st.secrets["grants"]["port"], 
                           dbname=st.secrets["grants"]["dbname"], 
                           user=st.secrets["grants"]["user"], 
                           password=st.secrets["grants"]["password"]
                           )
    cur = conn.cursor()
    #st.write(f"Executing query: {cur.mogrify(query, params).decode()}")  # Print the query with parameters inserted
    cur.execute(query, params)
    col_names = [desc[0] for desc in cur.description]
    results = pd.DataFrame(cur.fetchall(), columns=col_names)
    cur.close()
    conn.close()
    return results

def get_round_summary():
    sql_query_file = 'queries/get_rounds_summary_from_indexer.sql'
    with open(sql_query_file, 'r') as file:
        query = file.read()
    results = run_query(query)
    return results

def get_round_votes(round_id, chain_id):
    sql_query_file = 'queries/get_votes_by_round_id_from_indexer.sql'
    with open(sql_query_file, 'r') as file:
        query = file.read()
    params = {
        'round_id': round_id,
        'chain_id': chain_id
    }
    results = run_query_with_params(query, params)  # Ensure your run_query can handle parameterized inputs
    return results

def get_projects_in_round(round_id, chain_id):
    sql_query_file = 'queries/get_projects_summary_from_indexer.sql'
    with open(sql_query_file, 'r') as file:
        query = file.read()
    params = {
        'round_id': round_id,
        'chain_id': chain_id
    }
    results = run_query_with_params(query, params)
    return results

#@st.cache(ttl=3600, allow_output_mutation=True)
def get_token_price_from_dune(blockchain, token_address):
    DUNE_API_KEY = st.secrets['dune']['DUNE_API_KEY']
    sql_query_file = 'queries/get_token_price_from_dune.sql'
    with open(sql_query_file, 'r') as file:
            query = file.read()
    query = query.format(blockchain=blockchain, token_address=token_address)
    client = DuneClient(api_key=DUNE_API_KEY)
    results = client.run_sql(
        query_sql=query, 
        performance='large')
    data = results.result.rows
    df = pd.DataFrame(data)
    return df

@st.cache_data(ttl=3600) 
def load_data_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad responses
        return [json.loads(line) for line in response.text.splitlines()]
    except requests.RequestException as e:
        print(f"Failed to fetch data from {url}. Error: {e}")
        return []
    

def load_passport_model_scores(addresses):
    url = 'https://public.scorer.gitcoin.co/eth_model_scores_v2/eth_model_scores.jsonl'
    scores = load_data_from_url(url)
    scores = pd.DataFrame(scores)
    scores = scores.join(pd.json_normalize(scores['data'])).drop('data', axis=1)
    scores['address'] = scores['address'].str.lower()
    scores = scores[scores['address'].isin(addresses)]
    scores = scores.sort_values('updated_at', ascending=False).drop_duplicates('address')
    scores['score'] = scores['score'].astype(float)
    scores['rawScore'] = scores['score']

    df = pd.read_csv('data/gg20_missing_addresses_scores.csv')
    df.rename(columns={'scores': 'rawScore'}, inplace=True)
    df = df[df['address'].isin(addresses)]
    scores = scores[['address', 'rawScore']]
    df = pd.concat([df, scores], ignore_index=True)

    return df

def load_avax_scores(addresses):
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

def load_stamp_scores(addresses):
    addresses = tuple(addresses)
    sql_query_file = 'queries/get_passport_stamps.sql'
    with open(sql_query_file, 'r') as file:
        query = file.read()
    params = {
        'addresses': addresses
    }
    results = run_grants_query_with_params(query, params)  
    return results

def parse_config_file(file_content):
    data = []
    chain_pattern = re.compile(r'{\s*id:\s*(\d+),\s*name:\s*"([^"]+)",.*?tokens:\s*\[(.*?)\].*?}', re.DOTALL)
    token_pattern = re.compile(r'code:\s*"(?P<code>[^"]+)".*?address:\s*"(?P<address>[^"]+)".*?decimals:\s*(?P<decimals>\d+).*?priceSource:\s*{\s*chainId:\s*(?P<price_source_chain_id>\d+).*?address:\s*"(?P<price_source_address>[^"]+)"', re.DOTALL)
    chain_matches = chain_pattern.findall(file_content)
    print(f"Number of chain matches: {len(chain_matches)}")

    for chain_match in chain_matches:
        chain_id = int(chain_match[0])
        chain_name = chain_match[1]
        token_data = chain_match[2]

        #print(f"Chain ID: {chain_id}, Chain Name: {chain_name}")
        #print(f"Token Data: {token_data}")

        token_matches = token_pattern.finditer(token_data)

        for token_match in token_matches:
            #print(f"Token Match: {token_match.group()}")
            token_code = token_match.group('code')
            token_address = token_match.group('address')
            token_decimals = int(token_match.group('decimals'))
            price_source_chain_id = int(token_match.group('price_source_chain_id'))
            price_source_address = token_match.group('price_source_address')

            #print(f"Token Code: {token_code}, Token Address: {token_address}")

            data.append([
                chain_id,
                chain_name,
                token_code,
                token_address,
                token_decimals,
                price_source_chain_id,
                price_source_address
            ])
    #print(f"Data: {data}")
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
    
@st.cache_resource(ttl=3600)
def fetch_tokens_config():
    url = 'https://raw.githubusercontent.com/gitcoinco/grants-stack-indexer/main/src/config.ts'
    
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad responses
    except requests.RequestException as e:
        print(f"Failed to fetch data from {url}. Error: {e}")
        return None

    df = parse_config_file(response.text)
    return df


@st.cache_resource(ttl=3600)
def fetch_latest_price(chain_id, token_address, coingecko_api_key=st.secrets['coingecko']['COINGECKO_API_KEY'], coingecko_api_url="https://api.coingecko.com/api/v3"):
    platforms = {
        1: "ethereum",
        250: "fantom",
        10: "optimistic-ethereum",
        42161: "arbitrum-one",
        43114: "avalanche",
        713715: "sei-network",
    }

    native_tokens = {
        1: "ethereum",
        250: "fantom",
        10: "ethereum",
        42161: "ethereum",
        43114: "avalanche-2",
        713715: "sei-network",
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

