import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
import utils
import fundingutils


st.set_page_config(
    page_title="Matching Results",
    page_icon="favicon.png",
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
    
if st.session_state.round_id is None or st.session_state.chain_id is None:
    st.header("Oops! Something went wrong. You're not supposed to be here 🙈")
    st.subheader("Please provide round_id and chain_id in the URL")
    st.subheader('Example: https://qf-calculator.fly.dev/?round_id=23&chain_id=42161')
    st.stop()
else:
    round_id = st.session_state.round_id.lower()
    chain_id = int(st.session_state.chain_id)

st.image('657c7ed16b14af693c08b92d_GTC-Logotype-Dark.png', width = 300)

rounds = utils.get_round_summary()
rounds = rounds[(rounds['round_id'].str.lower() == round_id) & (rounds['chain_id'] == chain_id)]

round_name = rounds['round_name'].values[0]
matching_cap_amount = rounds['matching_cap_amount'].astype(float).values[0] if 'matching_cap_amount' in rounds and not pd.isnull(rounds['matching_cap_amount'].values[0]) else 100.0
matching_funds_available = rounds['matching_funds_available'].astype(float).values[0] if 'matching_funds_available' in rounds else 0
min_donation_threshold_amount = rounds['min_donation_threshold_amount'].astype(float).values[0] if 'min_donation_threshold_amount' in rounds and not pd.isnull(rounds['min_donation_threshold_amount'].values[0]) else 0.0
sybilDefense = rounds['sybilDefense'].values[0] if 'sybilDefense' in rounds and rounds['sybilDefense'].values[0] is not None else 'false'
token = rounds['token'].values[0] if 'token' in rounds else 'ETH'
chain = blockchain_mapping.get(rounds['chain_id'].values[0] if 'chain_id' in rounds else 1)

st.title(f'{round_name} - Matching Results')
matching_amount = rounds['matching_funds_available'].astype(float).values[0]
df = utils.get_round_votes(round_id, chain_id)


## LOAD PASSPORT DATA 
unique_voters = df['voter'].drop_duplicates()

if chain_id == 43114: 
    scores = utils.load_avax_scores(unique_voters)
    score_at_50_percent = 25
    score_at_100_percent = 25
    sybilDefense = 'Avalanche Passport'
elif sybilDefense == 'true':
    score_at_50_percent = 15
    score_at_100_percent = 25
    scores = utils.load_stamp_scores(unique_voters)
    sybilDefense = 'Passport Stamps'
else:
    score_at_50_percent = 1
    score_at_100_percent = 25
    scores = utils.load_passport_model_scores(unique_voters)
    sybilDefense = 'Passport Model Based Detection System'

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
col1.write(f"**Chain:** {chain}")
col1.write(f"**Matching Cap:** {matching_cap_amount:.2f}%")
col1.write(f"**Sybil Defense Selected:** {sybilDefense}")
total_voters = df['voter'].nunique()
col1.write(f"**Number of Unique Voters:** {total_voters}")
col2.write(f"**Matching Available:** {matching_funds_available:.2f}  {matching_token_symbol}")
col2.write(f"**Matching Token Price:** ${matching_token_price:.2f}")
col2.write(f"**Minimum Donation Threshold Amount:** ${min_donation_threshold_amount:.2f}")
col2.write(f"**Number of Unique Projects:** {df['project_name'].nunique()}")



if min_donation_threshold_amount == 1.0:
    min_donation_threshold_amount = 0.99

df = pd.merge(df, scores[['address', 'rawScore']], left_on='voter', right_on='address', how='left')



#turn_off_passport = st.sidebar.checkbox('Turn off passport', value=False)
#if turn_off_passport:
#    st.write('Passport is turned off')
#    score_at_50_percent = 0
#    score_at_100_percent = 0


st.header('🛂 Passport Usage')
st.subheader(f" {len(scores)} Users ({len(scores)/total_voters*100:.1f}%) Have a Passport Score")

def calculate_verified_vs_unverified(scores, donations_df, score_threshold):
    verified_users_count = scores[scores['rawScore'] >= score_threshold].shape[0]
    verified_funding_total = donations_df.loc[donations_df['rawScore'] >= score_threshold, 'amountUSD'].sum()
    unverified_users_count = scores[scores['rawScore'] < score_threshold].shape[0]
    unverified_funding_total = donations_df.loc[donations_df['rawScore'] < score_threshold, 'amountUSD'].sum()

    summary_data = pd.DataFrame({
        'Category': ['Verified', 'Unverified'],
        'Users': [verified_users_count, unverified_users_count],
        'Crowdfunding': [verified_funding_total, unverified_funding_total]
    })

    # Calculate percentages
    total_users_count = summary_data['Users'].sum()
    total_crowdfunding_amount = summary_data['Crowdfunding'].sum()
    summary_data['Percentage Of Users'] = (summary_data['Users'] / total_users_count) * 100
    summary_data['Percentage Of Crowdfunding'] = (summary_data['Crowdfunding'] / total_crowdfunding_amount) * 100

    # Create the plot
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=summary_data['Percentage Of Crowdfunding'],
            y=summary_data['Category'],
            orientation='h',
            name='Percentage Of Crowdfunding',
            text=summary_data['Percentage Of Crowdfunding'].apply(lambda x: f"{x:.1f}%"),
            textposition='auto',
            hovertemplate='<b>%{y}</b><br>Percentage Of Crowdfunding: %{x:.1f}%<br>Crowdfunding: $%{customdata:,.0f}<extra></extra>',
            customdata=summary_data['Crowdfunding']
        )
    )

    fig.add_trace(
        go.Bar(
            x=summary_data['Percentage Of Users'],
            y=summary_data['Category'],
            orientation='h',
            name='Percentage Of Users',
            text=summary_data['Percentage Of Users'].apply(lambda x: f"{x:.1f}%"),
            textposition='auto',
            hovertemplate='<b>%{y}</b><br>Percentage Of Users: %{x:.1f}%<br>Users: %{customdata:,}<extra></extra>',
            customdata=summary_data['Users']
        )
    )

    fig.update_layout(
        title_text='Percentage of Crowdfunding and Users by Passport Status',
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

    return fig, summary_data

# Call the function
passport_usage_fig, passport_usage_df = calculate_verified_vs_unverified(scores, df, score_at_50_percent)
st.plotly_chart(passport_usage_fig, use_container_width=True)
#st.write(passport_usage_df)


st.header('👥 Crowdfunding')
st.subheader(f"${'{:,.2f}'.format(df['amountUSD'].sum())} raised by crowd")
matching_amount_display = matching_token_price * matching_amount
crowd_raised = df['amountUSD'].sum()
percentage_of_matching = (crowd_raised / matching_amount_display) * 100
st.subheader(f"{percentage_of_matching:.2f}% of the matching pool")
st.write('')
# Group df by voter and calculate statistics
grouped_voter_data = df.groupby('voter')['amountUSD'].sum().reset_index()


avg_donation = grouped_voter_data['amountUSD'].mean()
median_donation = grouped_voter_data['amountUSD'].median()
max_donation = grouped_voter_data['amountUSD'].max()

# Display the statistics
col1, col2, col3 = st.columns(3)
col1.metric(label="Median Donor Contribution", value=f"${median_donation:.2f}")
col2.metric(label="Average Donor Contribution", value=f"${avg_donation:.2f}")
col3.metric(label="Max Donor Contribution", value=f"${max_donation:.2f}")


# Define the bin edges with smaller intervals for lower amounts
bin_edges = [0, 1, 2, 3, 4, 5, 10, 20, 30, 50, 100, 500, 1000, np.inf]
bin_labels = ['0-1','1-2','2-3','3-4', '4-5', '5-10', '10-20','20-30','30-50', '50-100', '100-500', '500-1000', '1000+']

# Assign each donation amount to a bin
grouped_voter_data['amountUSD_bin'] = pd.cut(grouped_voter_data['amountUSD'], bins=bin_edges, labels=bin_labels, right=False)

# Create a distribution chart of the grouped_voter_data with custom bins
fig = px.histogram(grouped_voter_data, x="amountUSD_bin", category_orders={'amountUSD_bin': bin_labels},
                   labels={'amountUSD_bin': 'Donation Amount Range (USD)'}, nbins=len(bin_edges)-1)

# Update hover text to be more descriptive
fig.update_traces(hovertemplate='<b>Donation Range:</b> $%{x}<br><b>Number of Donors:</b> %{y}')

fig.update_layout(
    title_text='Distribution of Donor Contributions by Amount',
    xaxis=dict(title='Donation Amount Range (USD)', titlefont=dict(size=18), tickfont=dict(size=14)),
    yaxis=dict(title='Number of Donors', titlefont=dict(size=18), tickfont=dict(size=14)),
    bargap=0.3  # add space between bars
)

st.plotly_chart(fig, use_container_width=True)


st.header('💚 Quadratic Funding Results Comparison')
st.write('''Quadratic funding helps us solve coordination failures by creating a way for community members to fund what matters to them while amplifying their impact. However, it's assumption that people make independent decisions can be exploited to unfairly influence the distribution of matching funds.

Connection-oriented cluster-matching (COCM) doesn’t make this assumption. Instead, it quantifies just how coordinated groups of actors are likely to be based on the social signals they have in common. Projects backed by more independent agents receive greater matching funds. Conversely, if a project’s support network shows higher levels of coordination, the matching funds are reduced, encouraging self-organized solutions within more coordinated groups.

''')


all_projects = df['project_name'].unique()
st.write('')
projects_to_remove = st.multiselect('Projects may be removed from the matching distribution by selecting them here:', all_projects)
df = df[~df['project_name'].isin(projects_to_remove)]
st.write('')

# Calculate matching results with passport on
df_with_passport = fundingutils.apply_voting_eligibility(df.copy(), min_donation_threshold_amount, score_at_50_percent, score_at_100_percent)
votes_df_with_passport = fundingutils.pivot_votes(df_with_passport)

# Calculate matching results with passport off
#df_without_passport = fundingutils.apply_voting_eligibility(df.copy(), min_donation_threshold_amount, 0, 0)
#votes_df_without_passport = fundingutils.pivot_votes(df_without_passport)
#suffixes = ['(Passport On)', '(Passport Off)']

def get_matching(strategy, votes_df, matching_amount, suffix=None):
    df = fundingutils.get_qf_matching(strategy, votes_df, matching_cap_amount, matching_amount,cluster_df=votes_df)
    df = df.rename(columns={'project_name': 'Project', 'matching_amount': f'{strategy} Match', 'matching_percent': f'{strategy} Match %'})
    return df

strategies = ['COCM', 'QF']

# Calculate matching results for both strategies and both scenarios
matching_dfs = []
#for suffix, votes_df in zip(suffixes, [votes_df_with_passport, votes_df_without_passport]):
for strategy in strategies:
    matching_dfs.append(get_matching(strategy, votes_df_with_passport, matching_amount))

# Merge all matching results into a single DataFrame
matching_df = matching_dfs[0]
for dft in matching_dfs[1:]:
    matching_df = pd.merge(matching_df, dft, on='Project', how='outer')

# Ensure there are no duplicate rows in the dataframe before merging
df_unique = df[['project_name', 'chain_id', 'round_id', 'application_id']].drop_duplicates()

matching_df = pd.merge(matching_df, df_unique, left_on='Project', right_on='project_name', how='left')
matching_df['Project Page'] = 'https://explorer.gitcoin.co/#/round/' + matching_df['chain_id'].astype(str) + '/' + matching_df['round_id'].astype(str) + '/' + matching_df['application_id'].astype(str)

# Sort the dataframe by 'COCM Match' in descending order
matching_df = matching_df.sort_values(by='COCM Match', ascending=False)

# Configure the dataframe display
column_config = {
    "Project": st.column_config.TextColumn("Project"),
    "COCM Match": st.column_config.NumberColumn("COCM Match", format="%.2f"),
    "QF Match": st.column_config.NumberColumn("QF Match", format="%.2f"),
    "COCM Match %": st.column_config.NumberColumn("COCM Match %", format="%.2f"),
    "QF Match %": st.column_config.NumberColumn("QF Match %", format="%.2f"),
    "Project Page": st.column_config.LinkColumn("Project Page", display_text="Visit")
}

# Reorder columns to ensure 'Project' is first and 'Project Page' is last
matching_df = matching_df.drop(columns=['project_name', 'chain_id', 'round_id', 'application_id'])
columns_order = ['Project'] + [col for col in matching_df.columns if col not in ['Project', 'Project Page']] + ['Project Page']
display_matching_df = matching_df[columns_order]

# Use Streamlit's dataframe to display the data
st.subheader('Matching Results')
st.dataframe(
    display_matching_df,
    use_container_width=True,
    column_config=column_config,
    hide_index=True
)

st.markdown('Matching Values shown above are in **' + matching_token_symbol + '**')
##

st.subheader('⬇️ Download Matching Distribution')
# Let the user pick COCM or QF
strategy_choice = st.selectbox(
    'Select the matching strategy to download:',
    ('COCM', 'QF')
)
st.write('You can upload this CSV to manager.gitcoin.co to apply the matching results to your round')

# Filter the matching_df based on the selected strategy
if strategy_choice == 'COCM':
    matching_df['Match'] = matching_df['COCM Match']
else:
    matching_df['Match'] = matching_df['QF Match']




output_df = matching_df[['Project', 'Match']]
projects_df = utils.get_projects_in_round(round_id, chain_id)
projects_df = projects_df[~projects_df['project_name'].isin(projects_to_remove)]



output_df = pd.merge(output_df, projects_df, left_on='Project', right_on='project_name', how='outer')
output_df = output_df.rename(columns={'id': 'applicationId', 'project_id':'projectId', 'project_name': 'projectName', 'recipient_address':'payoutAddress', 'total_donations_count':'contributionsCount', 'Match': 'matched', 'total_amount_donated_in_usd':'totalReceived'})
output_df = output_df[['applicationId', 'projectId', 'projectName', 'payoutAddress', 'matched', 'contributionsCount', 'totalReceived']]
output_df['matchedUSD'] = (output_df['matched'] * matching_token_price).round(2)
output_df['matched'] = (output_df['matched'] * 10**matching_token_decimals).astype(int)
output_df['totalReceived'] = output_df['totalReceived'] * (10**matching_token_decimals)
output_df = output_df.fillna(0)


full_matching_funds_available = int(matching_funds_available * 10**matching_token_decimals)
output_df['matched'] = (output_df['matched'] * ((full_matching_funds_available-10) / sum(output_df['matched']))).astype(int)
all_matching_funds_available = full_matching_funds_available  > (output_df['matched'].sum())+10
#st.header('Trying to match: ' + '{:.0f}'.format(output_df['matched'].sum()) + ' out of ' + '{:.0f}'.format(full_matching_funds_available))
while not all_matching_funds_available:
    full_matching_funds_available -= 10
    output_df['matched'] = (output_df['matched'] * ((full_matching_funds_available) / sum(output_df['matched']))).astype(int)
    all_matching_funds_available = full_matching_funds_available  > (output_df['matched'].sum())
    st.warning('The total matched funds exceed the available matching funds. Please talk to @umarkhaneth on telegram. \n'
               'Matching funds available: ' + '{:.0f}'.format(matching_funds_available * 10**matching_token_decimals) + '\n'
               'Total matched funds: ' + '{:.0f}'.format(output_df['matched'].sum()) + '\n'
               'Difference: ' + str((output_df['matched'].sum() - matching_funds_available * 10**matchisng_token_decimals)))
#st.header('Proceeding: ' + '{:.0f}'.format(output_df['matched'].sum()) + ' out of ' + '{:.0f}'.format(full_matching_funds_available))

# Add additional columns
#output_df['matched'] = output_df['matched'].apply(lambda x: '{:.0f}'.format(x) if pd.notnull(x) else x)   
output_df['totalReceived'] = output_df['totalReceived'].apply(lambda x: '{:.0f}'.format(x) if pd.notnull(x) else x)   


output_df['sumOfSqrt'] = 0
output_df['capOverflow'] = 0
output_df['matchedWithoutCap'] = 0
output_df = output_df[['applicationId', 'projectId', 'projectName', 'payoutAddress', 'matchedUSD', 'totalReceived', 'contributionsCount', 'matched', 'sumOfSqrt', 'capOverflow', 'matchedWithoutCap']]


st.write(output_df)
st.download_button(
    label="⬇ Download Matching Distribution",
    data=output_df.to_csv(index=False),
    file_name='output_data.csv',
    mime='text/csv'
)
