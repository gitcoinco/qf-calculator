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
matching_cap_amount = rounds['matching_cap_amount'].astype(float).values[0] if 'matching_cap_amount' in rounds and not pd.isnull(rounds['matching_cap_amount'].values[0]) else 100.0
matching_funds_available = rounds['matching_funds_available'].astype(float).values[0] if 'matching_funds_available' in rounds else 0
min_donation_threshold_amount = rounds['min_donation_threshold_amount'].astype(float).values[0] if 'min_donation_threshold_amount' in rounds and not pd.isnull(rounds['min_donation_threshold_amount'].values[0]) else 0.0
sybilDefense = rounds['sybilDefense'].values[0] if 'sybilDefense' in rounds else False
token = rounds['token'].values[0] if 'token' in rounds else 'ETH'
chain = blockchain_mapping.get(rounds['chain_id'].values[0] if 'chain_id' in rounds else 1)

st.title(f'{round_name}')

#st.write(rounds)

matching_amount = rounds['matching_funds_available'].astype(float).values[0]
df = utils.get_round_votes(round_id, chain_id)
#st.write(df)




## LOAD PASSPORT DATA 
unique_voters = df['voter'].drop_duplicates()


if chain_id == 43114: 
    scores = utils.load_avax_scores(unique_voters)
    score_at_50_percent = 25
    score_at_100_percent = 25
    st.header('Cluster Match Results Using Avalanche Passport')
elif sybilDefense == 'true':
    st.header('Cluster Match Results Using Passport Stamps')
    score_at_50_percent = 15
    score_at_100_percent = 25
    scores = utils.load_stamp_scores(unique_voters)
else:
    st.header('Cluster Match Results Using Passport Model Based Detection System')
    score_at_50_percent = 1
    score_at_100_percent = 25
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
total_voters = df['voter'].nunique()
col1.write(f"Number of Unique Voters: {total_voters}")
col2.write(f"Matching Available: {matching_funds_available:.2f}")
col2.write(f"Matching Token:  {matching_token_symbol}")
col2.write(f"Matching Token Price: ${matching_token_price:.2f}")
col2.write(f"Minimum Donation Threshold Amount: ${min_donation_threshold_amount:.2f}")
col1.write(f"Number of Unique Projects: {df['project_name'].nunique()}")
col2.write(f"Amount Raised by Crowd: ${df['amountUSD'].sum():.2f}")

if min_donation_threshold_amount == 1.0:
    min_donation_threshold_amount = 0.99

df = pd.merge(df, scores[['address', 'rawScore', 'CivicUniquenessPass', 'HolonymGovIdProvider']], left_on='voter', right_on='address', how='left')
turn_off_passport = st.sidebar.checkbox('Turn off passport', value=False)
if turn_off_passport:
    st.write('Passport is turned off')
    score_at_50_percent = 0
    score_at_100_percent = 0


## FOR AVALANCHE:
# Add Donations by Passport Category
# Make it easy to compare passport to no passport 
# 

st.header('🛂 Passport Usage')


# Count of distinct addresses for each rawScore
# Count of distinct addresses for each category


#
# GRAPH BELOW 
civic_only = df[(df['CivicUniquenessPass'] > 0) & (df['HolonymGovIdProvider'] == 0)]
holonym_only = df[(df['CivicUniquenessPass'] == 0) & (df['HolonymGovIdProvider'] > 0)]
both = df[(df['CivicUniquenessPass'] > 0) & (df['HolonymGovIdProvider'] > 0)]
neither = df[(df['CivicUniquenessPass'] == 0) & (df['HolonymGovIdProvider'] == 0)]

# Create a DataFrame
score_counts = pd.DataFrame({
    'Category': ['Civic Verified', 'Holonym Verified', 'Both', 'Unverified'],
    'Users': [civic_only['voter'].nunique(), holonym_only['voter'].nunique(), both['voter'].nunique(), neither['voter'].nunique()],
    'Crowdfunding': [civic_only['amountUSD'].sum(), holonym_only['amountUSD'].sum(), both['amountUSD'].sum(), neither['amountUSD'].sum()]
})

total_users = score_counts['Users'].sum()
total_crowdfunding = score_counts['Crowdfunding'].sum()

score_counts['Pct of Users'] = ((score_counts['Users'] / total_users) * 100).map("{:.2f}".format)
score_counts['Pct of Crowdfunding'] = ((score_counts['Crowdfunding'] / total_crowdfunding) * 100).map("{:.2f}".format)


verified_users = score_counts.loc[score_counts['Category'] != 'Unverified', 'Users'].sum()
verified_funding = score_counts.loc[score_counts['Category'] != 'Unverified', 'Crowdfunding'].sum()
unverified_users = score_counts.loc[score_counts['Category'] == 'Unverified', 'Users'].sum()
unverified_funding = score_counts.loc[score_counts['Category'] == 'Unverified', 'Crowdfunding'].sum()

combined_data = pd.DataFrame({
    'Category': ['Verified', 'Unverified'],
    'Users': [verified_users, unverified_users],
    'Crowdfunding': [verified_funding, unverified_funding]
})

# Calculate percentages
combined_data['Percentage Of Users'] = (combined_data['Users'] / total_users) * 100
combined_data['Percentage Of Crowdfunding'] = (combined_data['Crowdfunding'] / total_crowdfunding) * 100



# Create the plot
fig = go.Figure()

fig.add_trace(
    go.Bar(
        x=combined_data['Percentage Of Crowdfunding'],
        y=combined_data['Category'],
        orientation='h',
        name='Percentage Of Crowdfunding',
        text=combined_data['Percentage Of Crowdfunding'].apply(lambda x: f"{x:.1f}%"),
        textposition='auto',
        hovertemplate='<b>%{y}</b><br>Percentage Of Crowdfunding: %{x:.1f}%<br>Crowdfunding: $%{customdata:,.0f}<extra></extra>',
        customdata=combined_data['Crowdfunding']
    )
)

fig.add_trace(
    go.Bar(
        x=combined_data['Percentage Of Users'],
        y=combined_data['Category'],
        orientation='h',
        name='Percentage Of Users',
        text=combined_data['Percentage Of Users'].apply(lambda x: f"{x:.1f}%"),
        textposition='auto',
        hovertemplate='<b>%{y}</b><br>Percentage Of Users: %{x:.1f}%<br>Users: %{customdata:,}<extra></extra>',
        customdata=combined_data['Users']
    )
)

fig.update_layout(
    title_text='1/3 of Donors Verify with Passport but Drive Nearly Half of All Donations',
    title_font=dict(size=24),
    xaxis=dict(
        title='Percentage',
        titlefont=dict(size=18),
        tickfont=dict(size=14),
    ),
    yaxis=dict(
        title='Category',
        titlefont=dict(size=18),
        tickfont=dict(size=14),
    ),
    barmode='group',
    legend=dict(
        traceorder='reversed'
    ),
    hoverlabel=dict(
        bgcolor="white",
        font_size=16,
        font_family="Rockwell"
    )
)

st.plotly_chart(fig, use_container_width=True)

score_counts['Crowdfunding'] = score_counts['Crowdfunding'] .map("{:.2f}".format)
st.write(score_counts)


df = fundingutils.apply_voting_eligibility(df, min_donation_threshold_amount, score_at_50_percent, score_at_100_percent)
#st.write(df)
votes_df = fundingutils.pivot_votes(df)

def get_matching(strategy, votes_df, matching_amount):
    df = fundingutils.get_qf_matching(strategy, votes_df, matching_cap_amount, matching_amount, cluster_df = votes_df)
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
st.markdown('Matching Values shown above are in **' + matching_token_symbol + '**')

import plotly.graph_objects as go

# Prepare data
slopegraph_df = matching_df[['Project', 'QF Match', 'COCM Match']].melt('Project', var_name='Strategy', value_name='Match')

# Create figure
fig = go.Figure()

# Calculate slopes for each project
slopegraph_df['Slope'] = slopegraph_df.groupby('Project')['Match'].transform(lambda x: x.diff().iloc[-1])

# Get top 5 projects with biggest positive and negative slopes
top_positive_slopes = slopegraph_df.sort_values('Slope', ascending=False)['Project'].unique()[:5]
top_negative_slopes = slopegraph_df.sort_values('Slope')['Project'].unique()[:5]

# Define color schemes for positive and negative slopes
color_positive = ['green', 'lime', 'forestgreen', 'darkgreen', 'lightgreen']
color_negative = ['orange', 'darkorange', 'coral', 'orangered', 'lightsalmon']

# Add traces for each project with positive slope
for i, project in enumerate(top_positive_slopes):
    project_data = slopegraph_df[slopegraph_df['Project'] == project]
    fig.add_trace(go.Scatter(x=project_data['Strategy'], y=project_data['Match'], mode='lines+markers', name=project, line=dict(color=color_positive[i])))

# Add traces for each project with negative slope
for i, project in enumerate(top_negative_slopes):
    project_data = slopegraph_df[slopegraph_df['Project'] == project]
    fig.add_trace(go.Scatter(x=project_data['Strategy'], y=project_data['Match'], mode='lines+markers', name=project, line=dict(color=color_negative[i])))
# Set labels and title
fig.update_layout(
    title={
        'text': 'Top 10 Project Matching Differences',
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
    xaxis_title='Strategy',
    yaxis_title='Match',
    font=dict(
        family="Courier New, monospace",
        size=22,
        color="RebeccaPurple"
    )
)
fig.update_layout(yaxis=dict( automargin=True))


# Show plot
st.plotly_chart(fig)

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