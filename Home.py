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

st.title(f'{round_name} Cluster Match Results')

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
    st.write('Using Avalanche Passport')
elif sybilDefense == 'true':
    st.write('Using Passport Stamps')
    score_at_50_percent = 15
    score_at_100_percent = 25
    scores = utils.load_stamp_scores(unique_voters)
else:
    st.write('Using Passport Model Based Detection System')
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
col1.write(f"Users With Passport: {len(scores)} ({len(scores)/total_voters*100:.2f}%)")
col1.write(f"Users With Passing Passport: {len(scores[scores['rawScore'] >= score_at_50_percent])} ({len(scores[scores['rawScore'] >= score_at_50_percent])/total_voters*100:.2f}%)")
col2.write(f"Matching Available: {matching_funds_available:.2f}")
col2.write(f"Matching Token:  {matching_token_symbol}")
col2.write(f"Matching Token Price: ${matching_token_price:.2f}")
col2.write(f"Minimum Donation Threshold Amount: ${min_donation_threshold_amount:.2f}")
col2.write(f"Number of Unique Projects: {df['project_name'].nunique()}")

st.header('💚 Quadratic Funding Results Comparison')
st.write('''Quadratic funding helps us solve coordination failures by creating a way for community members to fund what matters to them while amplifying their impact. However, it's assumption that people make independent decisions can be exploited to unfairly influence the distribution of matching funds.

Connection-oriented cluster-matching (COCM) doesn’t make this assumption. Instead, it quantifies just how coordinated groups of actors are likely to be based on the social signals they have in common. Projects backed by more independent agents receive greater matching funds. Conversely, if a project’s support network shows higher levels of coordination, the matching funds are reduced, encouraging self-organized solutions within more coordinated groups.

''')

if min_donation_threshold_amount == 1.0:
    min_donation_threshold_amount = 0.99

df = pd.merge(df, scores[['address', 'rawScore']], left_on='voter', right_on='address', how='left')
<<<<<<< Updated upstream
#turn_off_passport = st.sidebar.checkbox('Turn off passport', value=False)
#if turn_off_passport:
#    st.write('Passport is turned off')
#    score_at_50_percent = 0
#    score_at_100_percent = 0
=======

st.header('🛂 Passport Usage')
>>>>>>> Stashed changes

# Create a histogram of rawScore by count of unique addresses
fig = px.histogram(df.drop_duplicates(subset='address'), x='rawScore', nbins=100, title='Distribution of Raw Scores',
                   labels={'rawScore': 'Raw Score', 'count': 'Count of Unique Addresses'})

# Update layout for better visualization
fig.update_layout(
    xaxis_title='Raw Score',
    yaxis_title='Count of Unique Addresses',
    bargap=0
)
# Display the plot
st.plotly_chart(fig)



# GRAPH BELOW 
# Create a DataFrame



verified_users = scores[scores['rawScore'] >= score_at_50_percent].shape[0]
verified_funding = df.loc[df['rawScore'] >= score_at_50_percent, 'amountUSD'].sum()
unverified_users = scores[scores['rawScore'] < score_at_50_percent].shape[0]
unverified_funding = df.loc[df['rawScore'] < score_at_50_percent, 'amountUSD'].sum()

combined_data = pd.DataFrame({
    'Category': ['Verified', 'Unverified'],
    'Users': [verified_users, unverified_users],
    'Crowdfunding': [verified_funding, unverified_funding]
})

# Calculate percentages
total_users = combined_data['Users'].sum()
total_crowdfunding = combined_data['Crowdfunding'].sum()
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
    bargap=0.3,
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
st.write(combined_data)



##
# Calculate matching results with passport on
df_with_passport = fundingutils.apply_voting_eligibility(df.copy(), min_donation_threshold_amount, score_at_50_percent, score_at_100_percent)
votes_df_with_passport = fundingutils.pivot_votes(df_with_passport)

st.header('Votes Data')

st.write(df_with_passport)


# Calculate matching results with passport off
df_without_passport = fundingutils.apply_voting_eligibility(df.copy(), min_donation_threshold_amount, 0, 0)
votes_df_without_passport = fundingutils.pivot_votes(df_without_passport)

def get_matching(strategy, votes_df, matching_amount, suffix):
    df = fundingutils.get_qf_matching(strategy, votes_df, matching_cap_amount, matching_amount, cluster_df=votes_df)
    df = df.rename(columns={'project_name': 'Project', 'matching_amount': f'{strategy} Match {suffix}', 'matching_percent': f'{strategy} Match % {suffix}'})
    return df

<<<<<<< Updated upstream
strategies = ['COCM',  'QF']#, 'donation_profile_clustermatch', 'pairwise']  # Add or remove strategies as needed

votes_df = fundingutils.pivot_votes(df)
voter_data = df.groupby('voter').agg({'project_name': 'nunique', 'amountUSD': 'sum'}).reset_index()
voter_data.columns = ['Voter', 'Number of Projects Picked', 'Sum of USD Picked']
=======
strategies = ['COCM', 'QF']
suffixes = ['(Passport On)', '(Passport Off)']
>>>>>>> Stashed changes

# Calculate matching results for both strategies and both scenarios
matching_dfs = []
for suffix, votes_df in zip(suffixes, [votes_df_with_passport, votes_df_without_passport]):
    for strategy in strategies:
        matching_dfs.append(get_matching(strategy, votes_df, matching_amount, suffix))

# Merge all matching results into a single DataFrame
matching_df = matching_dfs[0]
for dft in matching_dfs[1:]:
    matching_df = pd.merge(matching_df, dft, on='Project', how='outer')

# Ensure there are no duplicate rows in the dataframe before merging
df_unique = df[['project_name', 'chain_id', 'round_id', 'application_id']].drop_duplicates()

matching_df = pd.merge(matching_df, df_unique, left_on='Project', right_on='project_name', how='left')
matching_df['Project Page'] = 'https://explorer.gitcoin.co/#/round/' + matching_df['chain_id'].astype(str) + '/' + matching_df['round_id'].astype(str) + '/' + matching_df['application_id'].astype(str)

# Sort the dataframe by 'COCM Match (Passport On)' in descending order
matching_df = matching_df.sort_values(by='COCM Match (Passport On)', ascending=False)

# Configure the dataframe display
column_config = {
    "Project": st.column_config.TextColumn("Project"),
    "COCM Match (Passport On)": st.column_config.NumberColumn("COCM Match (Passport On)", format="%.2f"),
    "QF Match (Passport On)": st.column_config.NumberColumn("QF Match (Passport On)", format="%.2f"),
    "COCM Match % (Passport On)": st.column_config.NumberColumn("COCM Match % (Passport On)", format="%.2f%%"),
    "QF Match % (Passport On)": st.column_config.NumberColumn("QF Match % (Passport On)", format="%.2f%%"),
    "COCM Match (Passport Off)": st.column_config.NumberColumn("COCM Match (Passport Off)", format="%.2f"),
    "QF Match (Passport Off)": st.column_config.NumberColumn("QF Match (Passport Off)", format="%.2f"),
    "COCM Match % (Passport Off)": st.column_config.NumberColumn("COCM Match % (Passport Off)", format="%.2f%%"),
    "QF Match % (Passport Off)": st.column_config.NumberColumn("QF Match % (Passport Off)", format="%.2f%%"),
    "Project Page": st.column_config.LinkColumn("Project Page", display_text="Visit")
}

# Reorder columns to ensure 'Project' is first and 'Project Link' is last
columns_order = ['Project'] + [col for col in matching_df.columns if col not in ['Project', 'Project Page']] + ['Project Page']
matching_df = matching_df[columns_order]

# Use Streamlit's dataframe to display the data
st.dataframe(
    matching_df.drop(columns=['project_name', 'chain_id', 'round_id', 'application_id']),
    use_container_width=True,
    column_config=column_config,
    hide_index=True
)

st.markdown('Matching Values shown above are in **' + matching_token_symbol + '**')
##

output_df = matching_df[['Project', 'COCM Match (Passport On)']]


projects_df = utils.get_projects_in_round(round_id, chain_id)


output_df = pd.merge(output_df, projects_df, left_on='Project', right_on='project_name', how='outer')
output_df = output_df.rename(columns={'id': 'applicationId', 'project_id':'projectId', 'project_name': 'projectName', 'recipient_address':'payoutAddress', 'total_donations_count':'contributionsCount', 'COCM Match (Passport On)': 'matched', 'total_amount_donated_in_usd':'totalReceived'})
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

# Add a 'Results Summary' section including a dataframe that has: Project, Matching Funds, Crowdfunding (in USD), and Unique Voters

# Calculate the required columns
results_summary_df = output_df[['projectName', 'matched', 'totalReceived']].copy()

# Convert totalReceived back into USD
conversion_value = 1e18  # Replace this with the actual conversion value if different
results_summary_df['totalReceived'] = (results_summary_df['totalReceived'].astype(float) / conversion_value).round(2)
results_summary_df['matched'] = (results_summary_df['matched'].astype(float) / 10**matching_token_decimals).round(2)

# Calculate unique voters for each project
unique_voters_df = df.groupby('project_name')['voter'].nunique().reset_index()
unique_voters_df = unique_voters_df.rename(columns={'voter': 'Unique Voters', 'project_name': 'projectName'})

# Calculate unique voters with a score above the 50% threshold for each project
unique_voters_above_50_df = df[df['rawScore'] >= score_at_50_percent].groupby('project_name')['voter'].nunique().reset_index()
unique_voters_above_50_df = unique_voters_above_50_df.rename(columns={'voter': 'Passing Voters', 'project_name': 'projectName'})

# Merge unique voters into results_summary_df
results_summary_df = pd.merge(results_summary_df, unique_voters_df, on='projectName', how='left')
results_summary_df['Avg Matching Per Voter'] = results_summary_df['matched'] / results_summary_df['Unique Voters']
results_summary_df = pd.merge(results_summary_df, unique_voters_above_50_df, on='projectName', how='left')
results_summary_df['Percent Passing'] = results_summary_df['Passing Voters'] / results_summary_df['Unique Voters'] * 100

# Merge project page into results_summary_df
matching_df_links = matching_df[['Project', 'Project Page']]
results_summary_df = pd.merge(results_summary_df, matching_df_links, left_on='projectName', right_on='Project', how='left')
results_summary_df = results_summary_df.drop(columns=['Project'])
results_summary_df = results_summary_df.rename(columns={
    'projectName': 'Project',
    'matched': 'Matching Funds',
    'totalReceived': 'Crowdfunding (in USD)',
    'Unique Voters': 'Unique Voters',
    'Passing Voters': 'Passing Voters',
    'Percent Passing': 'Percent Passing',
    'Avg Matching Per Voter': 'Avg Matching Per Voter',
    'Project Page': 'Project Page'
})

column_config = {
    "Project": st.column_config.TextColumn("Project"),
    "Matching Funds": st.column_config.NumberColumn("Matching Funds", format="%.2f"),
    "Crowdfunding (in USD)": st.column_config.NumberColumn("Crowdfunding (in USD)", format="%.2f"),
    "Unique Voters": st.column_config.NumberColumn("Unique Voters"),
    "Passing Voters": st.column_config.NumberColumn("Passing Voters"),
    "Percent Passing": st.column_config.NumberColumn("Percent Passing", format="%.2f"),
    "Avg Matching Per Voter": st.column_config.NumberColumn("Avg Matching Per Voter", format="%.2f"),
    "Project Page": st.column_config.LinkColumn("Project Page")
}

# Display the 'Results Summary' section
st.header('Results Summary')
st.dataframe(results_summary_df, use_container_width=True, column_config=column_config, hide_index=True)