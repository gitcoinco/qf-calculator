import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
import utils
import fundingutils


st.set_page_config(
    page_title="Cluster Match Results",
    page_icon="📊",
    layout="wide",
)

blockchain_mapping = {
        1: "Ethereum",
        10: "Optimism",
        137: "Polygon",
        250: "Fantom",
        324: "ZKSync",
        8453: "Base",
        42161: "Arbitrum",
        43114: "Avalanche",
        534352: "Scroll"
    }
    
if 'round_id' not in st.session_state:
    st.session_state.round_id = None

if 'chain_id' not in st.session_state:
    st.session_state.chain_id = None
    
# Grab round_id from URL
query_params_round_id = st.query_params.get_all('round_id')
if len(query_params_round_id) == 1 and not st.session_state.round_id:
    st.session_state.round_id = query_params_round_id[0]

query_params_chain_id = st.query_params.get_all('chain_id')
if len(query_params_chain_id) == 1 and not st.session_state.chain_id:
    st.session_state.chain_id = query_params_chain_id[0]
    
round_id = st.session_state.round_id.lower()
chain_id = int(st.session_state.chain_id)

rounds = utils.get_round_summary()
#st.write(rounds)
rounds = rounds[(rounds['round_id'].str.lower() == round_id) & (rounds['chain_id'] == chain_id)]


round_name = rounds['round_name'].values[0]
matching_cap_amount = rounds['matching_cap_amount'].astype(float).values[0] if 'matching_cap_amount' in rounds and not pd.isnull(rounds['matching_cap_amount'].values[0]) else 'No Cap'
matching_funds_available = rounds['matching_funds_available'].astype(float).values[0] if 'matching_funds_available' in rounds else 0
min_donation_threshold_amount = rounds['min_donation_threshold_amount'].astype(float).values[0] if 'min_donation_threshold_amount' in rounds and not pd.isnull(rounds['min_donation_threshold_amount'].values[0]) else 0.0
sybilDefense = rounds['sybilDefense'].values[0] if 'sybilDefense' in rounds else False
token = rounds['token'].values[0] if 'token' in rounds else 'ETH'
chain = blockchain_mapping.get(rounds['chain_id'].values[0] if 'chain_id' in rounds else 1)

st.title(f'{round_name} Cluster Match Results')

#st.write(rounds)

matching_amount = rounds['matching_funds_available'].astype(float).values[0]
df = utils.get_round_votes(round_id, chain_id)
#st.write(df)




## LOAD PASSPORT DATA 
unique_voters = df['voter'].drop_duplicates()

if chain_id == 43114: 
    scores = utils.load_avax_scores(unique_voters)
    st.write('Using Avalanche Passport')
elif sybilDefense == 'true':
    st.write('Using Passport Stamps')
    scores = utils.load_stamp_scores(unique_voters)
else:
    st.write('Using Passport Model Based Detection System')
    scores = utils.load_passport_model_scores(unique_voters)


## LOAD TOKEN DATA 
config_df = utils.fetch_tokens_config()

config_df = config_df[(config_df['chain_id'] == chain_id) & (config_df['token_address'] == token)]
price_source_chain_id = config_df['price_source_chain_id'].iloc[0]
price_source_token_address = config_df['price_source_address'].iloc[0]
matching_token_decimals = config_df['token_decimals'].iloc[0]
matching_token_symbol = config_df['token_code'].iloc[0]

with st.spinner('Fetching token price...'):
    matching_token_price = utils.fetch_latest_price(price_source_chain_id, price_source_token_address)

st.header('⚙️ Round Settings')
col1, col2 = st.columns(2)
col1.write(f"Chain: {chain}")
col1.write(f"Matching Cap: {matching_cap_amount:.2f}%")
col1.write(f"Gitcoin Passport Used: {sybilDefense.capitalize()}")
col1.write(f"Number of Unique Voters: {df['voter'].nunique()}")
col1.write(f"Users With Passport: {len(scores)}")

col2.write(f"Matching Available: {matching_funds_available:.2f}")
col2.write(f"Matching Token:  {matching_token_symbol}")
col2.write(f"Matching Token Price: ${matching_token_price:.2f}")
col2.write(f"Minimum Donation Threshold Amount: ${min_donation_threshold_amount:.2f}")
col2.write(f"Number of Unique Projects: {df['project_name'].nunique()}")




votes_df = fundingutils.pivot_votes(df)

def get_matching(strategy, votes_df, matching_amount):
    df = fundingutils.get_qf_matching(strategy, votes_df, 100, matching_amount, cluster_df = votes_df)
    df = df.rename(columns={'project_name': 'Project', 'matching_amount': f'{strategy} Match', 'matching_percent': f'{strategy} Match %'})
    return df

strategies = ['COCM',  'QF']#, 'donation_profile_clustermatch', 'pairwise']  # Add or remove strategies as needed

votes_df = fundingutils.pivot_votes(df)
voter_data = df.groupby('voter').agg({'project_name': 'nunique', 'amountUSD': 'sum'}).reset_index()
voter_data.columns = ['Voter', 'Number of Projects Picked', 'Sum of USD Picked']


matching_dfs = [get_matching(strategy, votes_df, matching_amount) for strategy in strategies]

matching_df = matching_dfs[0]
for df in matching_dfs[1:]:
    matching_df = pd.merge(matching_df, df, on='Project', how='outer')


st.header('💚 Quadratic Funding Results Comparison')
st.write('''Quadratic funding helps us solve coordination failures by creating a way for community members to fund what matters to them while amplifying their impact. However, it's assumption that people make independent decisions can be exploited to unfairly influence the distribution of matching funds.

Collusion-oriented cluster-matching (COCM) doesn’t make this assumption. Instead, it quantifies just how coordinated groups of actors are likely to be based on the social signals they have in common. Projects backed by more independent agents receive greater matching funds. Conversely, if a project’s support network shows higher levels of coordination, the matching funds are reduced, encouraging self-organized solutions within more coordinated groups.

''')


if 'QF' in strategies:
    for strategy in strategies:
        if strategy != 'QF':
            matching_df[f'{strategy} Diff'] = ( matching_df[f'{strategy} Match'] - matching_df['QF Match'] )
            
            st.metric(label=f"Matching Funds Redistributed by {strategy}", value=f"{matching_df[f'{strategy} Diff'].abs().sum():.2f}" + ' ' + matching_token_symbol)
            st.metric(label=f"Percentage of Matching Funds Redistributed by {strategy}", value=f"{matching_df[f'{strategy} Diff'].abs().sum() / matching_amount * 100:.2f}" + '%')


for column in matching_df.columns:
    if column != 'Project':
        matching_df[column] = matching_df[column].apply(lambda x: round(x, 2))

st.dataframe(matching_df, use_container_width=True)
st.write('Values shown in the table are in ' + matching_token_symbol)
output_df = matching_df[['Project', 'COCM Match']]


projects_df = utils.get_projects_in_round(round_id, chain_id)


output_df = pd.merge(output_df, projects_df, left_on='Project', right_on='project_name', how='outer')
output_df = output_df.rename(columns={'id': 'applicationId', 'project_id':'projectId', 'project_name': 'projectName', 'recipient_address':'payoutAddress', 'total_donations_count':'contributionsCount', 'COCM Match': 'matched', 'total_amount_donated_in_usd':'totalReceived'})
output_df = output_df[['applicationId', 'projectId', 'projectName', 'payoutAddress', 'matched', 'contributionsCount', 'totalReceived']]
output_df['matchedUSD'] = (output_df['matched'] * matching_token_price).round(2)
output_df['matched'] = output_df['matched'] * 10**matching_token_decimals
output_df['totalReceived'] = output_df['totalReceived'] * (1e18) # come back and put in round token

# Add additional columns
output_df['matched'] = output_df['matched'].apply(lambda x: '{:.0f}'.format(x) if pd.notnull(x) else x)   
output_df['totalReceived'] = output_df['totalReceived'].apply(lambda x: '{:.0f}'.format(x) if pd.notnull(x) else x)   


output_df['sumOfSqrt'] = 0
output_df['capOverflow'] = 0
output_df['matchedWithoutCap'] = 0
output_df = output_df[['applicationId', 'projectId', 'projectName', 'payoutAddress', 'matchedUSD', 'totalReceived', 'contributionsCount', 'matched', 'sumOfSqrt', 'capOverflow', 'matchedWithoutCap']]

st.header('Download COCM Matching Results')
st.write(output_df)
st.download_button(
    label="Download Output Data",
    data=output_df.to_csv(index=False),
    file_name='output_data.csv',
    mime='text/csv'
)
st.write('You can upload this CSV to manager.gitcoin.co to apply the cluster matching results to your round')