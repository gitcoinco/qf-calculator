import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timezone
import psycopg2 as pg
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

def get_round_summary():
    sql_query_file = 'queries/get_rounds_summary_from_indexer.sql'
    with open(sql_query_file, 'r') as file:
        query = file.read()
    results = run_query(query)
    return results

def get_round_votes(round_address):
    sql_query_file = 'queries/get_votes_by_round_id_from_indexer.sql'
    with open(sql_query_file, 'r') as file:
        query = file.read()
    query = query.format(round_address=round_address)
    results = run_query(query)
    return results

def get_projects_in_round(round_address):
    sql_query_file = 'queries/get_projects_summary_from_indexer.sql'
    with open(sql_query_file, 'r') as file:
        query = file.read()
    query = query.format(round_address=round_address)
    results = run_query(query)
    return results

#@st.cache(ttl=3600,  hash_funcs={DuneClient: id})
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