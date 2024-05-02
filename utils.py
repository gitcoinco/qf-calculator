import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timezone
import psycopg2 as pg
import json
from dune_client.types import QueryParameter
from dune_client.client import DuneClient



@st.cache_resource(ttl=36000)
def run_query(query):
    """Run query and return results"""
    try:
        conn = pg.connect(host=st.secrets["database"]["host"], 
                           port=st.secrets["database"]["port"], 
                           dbname=st.secrets["database"]["dbname"], 
                           user=st.secrets["database"]["user"], 
                           password=st.secrets["database"]["password"])
        cur = conn.cursor()
        cur.execute(query)
        col_names = [desc[0] for desc in cur.description]
        results = pd.DataFrame(cur.fetchall(), columns=col_names)
    except pg.Error as e:
        st.warning(f"ERROR: Could not execute the query. {e}")
    finally:
        conn.close()
    return results

def run_query_with_params(query, params):
    conn = pg.connect(host=st.secrets["database"]["host"], 
                           port=st.secrets["database"]["port"], 
                           dbname=st.secrets["database"]["dbname"], 
                           user=st.secrets["database"]["user"], 
                           password=st.secrets["database"]["password"])
    cur = conn.cursor()
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
    url = 'https://public.scorer.gitcoin.co/eth_model_scores/eth_model_scores.jsonl'
    scores = load_data_from_url(url)
    scores = pd.DataFrame(scores)
    scores = scores.join(pd.json_normalize(scores['data'])).drop('data', axis=1)
    scores = scores[scores['address'].isin(addresses)]
    scores = scores.sort_values('updated_at', ascending=False).drop_duplicates('address')
    return scores