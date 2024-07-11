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

def load_scores_and_set_defense(chain_id, sybilDefense, unique_voters):
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
    elif sybilDefense == 'passport-mbds':
        score_at_50_percent = 1
        score_at_100_percent = 25
        scores = utils.load_passport_model_scores(unique_voters)
        sybilDefense = 'Passport Model Based Detection System'
    else:
        score_at_50_percent = 0
        score_at_100_percent = 0
        scores = pd.DataFrame()
        scores['address'] = unique_voters
        scores['rawScore'] = 1
        sybilDefense = 'None'
    return scores, score_at_50_percent, score_at_100_percent, sybilDefense

def load_data(csv=None): 
    blockchain_mapping = {
            1: "Ethereum",
            10: "Optimism",
            137: "Polygon",
            250: "Fantom",
            324: "ZKSync",
            8453: "Base",
            42161: "Arbitrum",
            43114: "Avalanche",
            534352: "Scroll",
            1329: "SEI"
        }
    rounds = utils.get_round_summary()
    rounds = rounds[(rounds['round_id'].str.lower() == round_id) & (rounds['chain_id'] == chain_id)]
    token = rounds['token'].values[0] if 'token' in rounds else 'ETH'
    sybilDefense = rounds['sybil_defense'].values[0] if 'sybil_defense' in rounds else 'None'
    df = utils.get_round_votes(round_id, chain_id)
    if csv is not None:
        csv['address'] = csv['address'].str.lower()
        df['voter'] = df['voter'].str.lower()
        df['flagged'] = df['voter'].isin(csv['address'])
        flagged_votes_count = df["flagged"].sum()
        unique_flagged_voters_count = df[df['flagged']]['voter'].nunique()
        
        st.markdown(f"**Flagged Votes:** {flagged_votes_count}")
        st.markdown(f"**Unique Flagged Voters:** {unique_flagged_voters_count}")
        filter_flagged = st.toggle('Filter out flagged votes')
        if filter_flagged:
            df = df[~df['flagged']]
    config_df = utils.fetch_tokens_config()
    config_df = config_df[(config_df['chain_id'] == chain_id) & (config_df['token_address'] == token)]
    matching_token_price = utils.fetch_latest_price(config_df['price_source_chain_id'].iloc[0], config_df['price_source_address'].iloc[0])
    
    #df = df[df['voter'] != '0x38467762af703e60b57d99b87543f07ef2b3319c']
    
    unique_voters = df['voter'].drop_duplicates()
    scores, score_at_50_percent, score_at_100_percent, sybilDefense = load_scores_and_set_defense(chain_id, sybilDefense, unique_voters)

    return {
        "blockchain_mapping": blockchain_mapping,
        "rounds": rounds,
        "df": df,
        "config_df": config_df,
        "matching_token_price": matching_token_price,
        "scores": scores,
        "score_at_50_percent": score_at_50_percent,
        "score_at_100_percent": score_at_100_percent,
        "sybilDefense": sybilDefense
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
with st.expander("Advanced: Filter Out Wallets"):
    st.write('Upload a CSV file with a single column named "address" containing the ETH addresses to filter out. Addresses should include the 0x prefix.')
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
    if uploaded_file is not None:
        csv = pd.read_csv(uploaded_file)
        st.write("CSV file uploaded successfully. Here's a preview:")
        st.write(csv.head())
        data = load_data(csv)
    else:
        data = load_data()

blockchain_mapping = data["blockchain_mapping"]
rounds = data["rounds"]
df = data["df"]
config_df = data["config_df"]
matching_token_price = data["matching_token_price"]
scores = data["scores"]
score_at_50_percent = data["score_at_50_percent"]
score_at_100_percent = data["score_at_100_percent"]
sybilDefense = data["sybilDefense"]

round_name = rounds['round_name'].values[0]
matching_cap_amount = rounds['matching_cap_amount'].astype(float).values[0] if 'matching_cap_amount' in rounds and not pd.isnull(rounds['matching_cap_amount'].values[0]) else 100.0
matching_funds_available = rounds['matching_funds_available'].astype(float).values[0] if 'matching_funds_available' in rounds else 0
min_donation_threshold_amount = rounds['min_donation_threshold_amount'].astype(float).values[0] if 'min_donation_threshold_amount' in rounds and not pd.isnull(rounds['min_donation_threshold_amount'].values[0]) else 0.0
token = rounds['token'].values[0] if 'token' in rounds else 'ETH'
chain = blockchain_mapping.get(rounds['chain_id'].values[0] if 'chain_id' in rounds else 1)


st.title(f'{round_name} - Matching Results')

matching_amount = rounds['matching_funds_available'].astype(float).values[0]

## LOAD TOKEN DATA 

price_source_chain_id = config_df['price_source_chain_id'].iloc[0]
price_source_token_address = config_df['price_source_address'].iloc[0]
matching_token_decimals = config_df['token_decimals'].iloc[0]
matching_token_symbol = config_df['token_code'].iloc[0]


st.header('⚙️ Round Settings')
col1, col2 = st.columns(2)
col1.write(f"**Chain:** {chain}")
col1.write(f"**Matching Cap:** {matching_cap_amount:.2f}%")
col1.write(f"**Passport Defense Selected:** {sybilDefense}")
total_voters = df['voter'].nunique()
col1.write(f"**Number of Unique Voters:** {total_voters}")
col2.write(f"**Matching Available:** {matching_funds_available:.2f}  {matching_token_symbol}")
col2.write(f"**Matching Token Price:** ${matching_token_price:.2f}")
col2.write(f"**Minimum Donation Threshold Amount:** ${min_donation_threshold_amount:.2f}")
col2.write(f"**Number of Unique Projects:** {df['project_name'].nunique()}")



if min_donation_threshold_amount == 1.0:
    min_donation_threshold_amount = 0.99

df = pd.merge(df, scores[['address', 'rawScore']], left_on='voter', right_on='address', how='left')




def create_treemap(dfv):
    votes_by_voter_and_project = dfv.groupby(['voter', 'project_name'])['amountUSD'].sum().reset_index()
    votes_by_voter_and_project['voter'] = votes_by_voter_and_project['voter'].str[:10] + '...'
    votes_by_voter_and_project['shortened_title'] = votes_by_voter_and_project['project_name'].str[:15] + '...'
    
    fig = px.treemap(votes_by_voter_and_project, path=['shortened_title', 'voter'], values='amountUSD', hover_data=['project_name', 'amountUSD'])
    # Update hovertemplate to format the hover information
    fig.update_traces(
        texttemplate='%{label}<br>$%{value:.3s}',
        hovertemplate='<b>%{customdata[0]}</b><br>Amount: $%{customdata[1]:,.2f}',
        textposition='middle center',
        textfont_size=20
    )
    fig.update_traces(texttemplate='%{label}<br>$%{value:.3s}', textposition='middle center', textfont_size=20)
    fig.update_layout(font=dict(size=20))
    fig.update_layout(height=550)
    fig.update_layout(title_text="Donations by Grant")
    
    return fig

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
if sybilDefense != 'None':
    st.header('🛂 Passport Usage')
    st.subheader(f" {len(scores)} Users ({len(scores)/total_voters*100:.1f}%) Have a Passport Score")
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


st.header('💚 Quadratic Funding Method Comparison')
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




def get_matching(strategy, votes_df, matching_amount, suffix=None):
    df = fundingutils.get_qf_matching(strategy, votes_df, matching_cap_amount, matching_amount,cluster_df=votes_df)
    df = df.rename(columns={'project_name': 'Project', 'matching_amount': f'{strategy} Match'})
    return df[['Project', f'{strategy} Match']]

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
matching_df['Δ Match'] = matching_df['COCM Match'] - matching_df['QF Match']
# Configure the dataframe display
column_config = {
    "Project": st.column_config.TextColumn("Project"),
    "COCM Match": st.column_config.NumberColumn("COCM Match", format="%.2f"),
    "QF Match": st.column_config.NumberColumn("QF Match", format="%.2f"),
    "Δ Match": st.column_config.NumberColumn("Δ Match", format="%.2f"),
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




# FOCUS ON WHAT HAPPENS BELOW THIS POINT FOR DEBUGGING 



st.subheader('Select Matching Algorithm')
# Let the user pick COCM or QF
strategy_choice = st.selectbox(
    'Select the matching strategy to download:',
    ('COCM', 'QF')
)

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
output_df['matched'] = output_df['matched'].apply(lambda x: int(x * 10**matching_token_decimals))
output_df['totalReceived'] = output_df['totalReceived'].apply(lambda x: int(x * 10**matching_token_decimals))
output_df = output_df.fillna(0)

# Function to convert large integers to strings for display
def int_to_str(x):
    if isinstance(x, int) and x > 2**53 - 1:  # JavaScript's max safe integer
        return str(x)
    return x

# Convert large integers to strings for display
display_df = output_df.applymap(int_to_str)


full_matching_funds_available = int(matching_funds_available * 10**matching_token_decimals)
matching_overflow = output_df['matched'].sum() - full_matching_funds_available
if matching_overflow > 0:
    matching_adjustment = int(matching_overflow / output_df['matched'].count()+1)
    output_df['matched'] = output_df['matched']-matching_adjustment
    output_df['matched'] = output_df['matched'].apply(lambda x: int(x))


# SAFETY CHECK
all_matching_funds_are_available = full_matching_funds_available  > (output_df['matched'].sum())
if not all_matching_funds_are_available:
    st.warning('The total matched funds exceed the available matching funds. Please talk to @umarkhaneth on telegram. \n'
                'Matching funds available: ' + str(full_matching_funds_available) + '\n'
               'Total matched funds: ' + str(output_df['matched'].sum()) + '\n'
               'Difference: ' + str((output_df['matched'].sum() - full_matching_funds_available)))
    output_df['matched'] = output_df['matched']-1
    all_matching_funds_are_available = full_matching_funds_available  > (output_df['matched'].sum())


output_df['sumOfSqrt'] = 0
output_df['capOverflow'] = 0
output_df['matchedWithoutCap'] = 0
output_df = output_df[['applicationId', 'projectId', 'projectName', 'payoutAddress', 'matchedUSD', 'totalReceived', 'contributionsCount', 'matched', 'sumOfSqrt', 'capOverflow', 'matchedWithoutCap']]

display_df = output_df.applymap(int_to_str)
st.write(display_df)
st.download_button(
    label="⬇ Download Matching Distribution",
    data=output_df.to_csv(index=False),
    file_name=f'matching_distribution.csv',
    mime='text/csv'
)
st.write('You can upload this CSV to manager.gitcoin.co to apply the matching results to your round')
st.write('')
st.header('📈 Sharable Summary ')

# Create a summary table with initial columns
summary_df = output_df[['projectName', 'matchedUSD']].copy()


# Calculate unique voters and sum of crowdfunding
voter_stats = df.groupby('project_name').agg(
    uniqueVoters=('voter', 'nunique'),
    crowdfundingUSD=('amountUSD', 'sum')
).reset_index().rename(columns={'project_name': 'projectName'})

# Merge unique voters into summary_df
summary_df = summary_df.merge(voter_stats, on='projectName', how='left')
summary_df = summary_df.merge(matching_df[['Project', 'Match', 'Project Page']], left_on='projectName', right_on='Project', how='left')
summary_df.drop(columns=['Project'], inplace=True)
# Calculate average matching per voter
summary_df['averageMatchingPerVoter'] = summary_df['matchedUSD'] / summary_df['uniqueVoters']

# Rename columns for clarity and order them correctly
summary_df = summary_df.rename(columns={
    'projectName': 'Project',
    'Match': 'Matching Funds (' + matching_token_symbol + ')',
    'matchedUSD': 'Matching Funds (USD)',
    'crowdfundingUSD': 'Crowdfunding (USD)',
    'uniqueVoters': 'Unique Voters',
    'averageMatchingPerVoter': 'Average Matching Per Voter (USD)',
})[['Project', 'Matching Funds (' + matching_token_symbol + ')', 'Matching Funds (USD)', 'Crowdfunding (USD)', 'Unique Voters', 'Average Matching Per Voter (USD)', 'Project Page']]


# Ensure all USD columns have exactly two digits after the decimal
usd_columns = ['Matching Funds (USD)', 'Crowdfunding (USD)', 'Average Matching Per Voter (USD)', 'Matching Funds (' + matching_token_symbol + ')']
for col in usd_columns:
    summary_df[col] = summary_df[col].map(lambda x: round(x, 2))
summary_df = summary_df.sort_values('Matching Funds (' + matching_token_symbol + ')', ascending=True)
st.write(summary_df)
st.download_button(
    label="⬇ Download Summary",
    data=summary_df.to_csv(index=False),
    file_name=f'round_summary.csv',
    mime='text/csv'
)

# Limit to top 10 projects by matchedUSD
matching_by_projects = summary_df

# Create a pretty horizontal bar graph of matchedUSD and projectName
fig = px.bar(
    matching_by_projects,
    x='Matching Funds (' + matching_token_symbol + ')',
    y='Project',
    orientation='h',
    title='Matching Funds Distribution',
    labels={'Matching Funds (' + matching_token_symbol + ')': 'Matched Funds', 'Project': 'Project'},
    text=matching_by_projects['Matching Funds (' + matching_token_symbol + ')'].apply(lambda x: f"{x/1000:.1f}k" if x >= 1000 else f"{x:.0f}")
)

# Update layout for better appearance
fig.update_layout(
    xaxis_title='',
    yaxis_title='',
    yaxis=dict(tickmode='linear'),
    template='plotly_white',
    height=1640,
    width=800
)

# Display the bar graph
st.plotly_chart(fig)
