import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
import utils
import fundingutils

# Page configuration
st.set_page_config(page_title="Matching Results", page_icon="assets/favicon.png", layout="wide")

# Initialize session state variables
if 'round_id' not in st.session_state:
    st.session_state.round_id = None
if 'chain_id' not in st.session_state:
    st.session_state.chain_id = None

# Handle URL parameters for round_id and chain_id
query_params_round_id = st.query_params.get_all('round_id')
if len(query_params_round_id) == 1 and not st.session_state.round_id:
    st.session_state.round_id = query_params_round_id[0]

query_params_chain_id = st.query_params.get_all('chain_id')
if len(query_params_chain_id) == 1 and not st.session_state.chain_id:
    st.session_state.chain_id = query_params_chain_id[0]

def display_recent_rounds():
    # Fetch and process round data
    rounds = utils.get_round_summary()
    current_time = pd.Timestamp.now(tz='UTC')
    rounds = rounds[(rounds['donations_end_time'].dt.tz_convert('UTC') < current_time) & (rounds['votes'] > 0)]
    rounds = rounds.sort_values('donations_end_time', ascending=False)

    # Create round links and prepare display data
    rounds['Round Link'] = rounds.apply(lambda row: f"https://qf-calculator.fly.dev/?round_id={row['round_id']}&chain_id={row['chain_id']}", axis=1)
    rounds_display = rounds[['round_name', 'Round Link', 'votes', 'uniqueContributors', 'amountUSD']]
    
    # Configure column display
    column_config = {
        "Round Link": st.column_config.LinkColumn(
            "Round Link",
            display_text="Go to Round",
            help="Click to view round details"
        ),
        "votes": st.column_config.NumberColumn(
            "Total Votes",
            help="Total number of votes in the round"
        ),
        "uniqueContributors": st.column_config.NumberColumn(
            "Unique Contributors",
            help="Number of unique contributors in the round"
        ),
        "amountUSD": st.column_config.NumberColumn(
            "Total Amount (USD)",
            help="Total amount donated in USD",
            format="$%.2f"
        )
    }

    # Display the dataframe
    st.header("Recent Rounds That Have Ended:")
    st.dataframe(
        rounds_display.head(20),
        column_config=column_config,
        hide_index=True
    )


def validate_input():
    """Validate the presence of round_id and chain_id in the URL."""
    if st.session_state.round_id is None or st.session_state.chain_id is None:
        st.header("Oops! Something went wrong. You're not supposed to be here 🙈")
        st.subheader("Please provide round_id and chain_id in the URL")
        st.subheader('Example: https://qf-calculator.fly.dev/?round_id=23&chain_id=42161')
        display_recent_rounds()
        st.stop()
    return st.session_state.round_id.lower(), int(st.session_state.chain_id)

@st.cache_resource(ttl=36000)
def load_scores_and_set_defense(chain_id, sybilDefense, unique_voters):
    """Load scores and set Sybil defense parameters based on chain and defense type."""
    if chain_id == 43114:  # AVALANCHE 
        scores = utils.load_avax_scores(unique_voters)
        score_at_50_percent = score_at_100_percent = 25
        sybilDefense = 'Avalanche Passport'
    elif sybilDefense == 'true': 
        scores = utils.load_stamp_scores(unique_voters)
        score_at_50_percent, score_at_100_percent = 15, 25
        sybilDefense = 'Passport Stamps'
    elif sybilDefense == 'passport-mbds':
        scores = utils.load_passport_model_scores(unique_voters)
        score_at_50_percent, score_at_100_percent = 25,50
        sybilDefense = 'Passport Model Based Detection System'
    else:
        # If no Sybil defense is set, assign a default score of 1 to all voters
        scores = pd.DataFrame({'address': unique_voters, 'rawScore': 1})
        score_at_50_percent = score_at_100_percent = 0
        sybilDefense = 'None'
    return scores, score_at_50_percent, score_at_100_percent, sybilDefense

def load_data(round_id, chain_id, filter_in_csv=None, filter_out_csv=None):
    """Load and process data for the specified round and chain."""
    blockchain_mapping = {1: "Ethereum", 10: "Optimism", 137: "Polygon", 250: "Fantom",
                          324: "ZKSync", 8453: "Base", 42161: "Arbitrum", 43114: "Avalanche",
                          534352: "Scroll", 1329: "SEI"}
    rounds = utils.get_round_summary()
    
    # """
    # TESTING
    # """
    # rt = rounds[rounds['sybilDefense'] == 'passport-mbds']
    # st.write(rt[rt['uniqueContributors'] > 0].head(20))

    rounds = rounds[(rounds['round_id'].str.lower() == round_id) & (rounds['chain_id'] == chain_id)] # FILTER BY ROUND_ID AND CHAIN_ID
    
    token = rounds['token'].values[0] if 'token' in rounds else 'ETH'
    sybilDefense = rounds['sybilDefense'].values[0] if 'sybilDefense' in rounds else 'None'
    df = utils.get_round_votes(round_id, chain_id)
    
    if filter_out_csv is not None:
        df = process_filterout_csv(df, filter_out_csv)
    
    if filter_in_csv is not None:
        filter_in_list = list(filter_in_csv['address'].str.lower())
    else:
        filter_in_list = []

    # Fetch token configuration and price
    config_df = utils.fetch_tokens_config()
    config_df = config_df[(config_df['chain_id'] == chain_id) & (config_df['token_address'] == token)]
    matching_token_price = utils.fetch_latest_price(config_df['price_source_chain_id'].iloc[0], config_df['price_source_address'].iloc[0])
    
    unique_voters = df['voter'].unique()

    scores, score_at_50_percent, score_at_100_percent, sybilDefense = load_scores_and_set_defense(chain_id, sybilDefense, unique_voters)

    # Merge scores with the main dataframe
    df = pd.merge(df, scores[['address', 'rawScore']], left_on='voter', right_on='address', how='left')
    df['rawScore'] = df['rawScore'].fillna(0)  # Fill NaN values with 0 for voters without a score
    
    return {
        "blockchain_mapping": blockchain_mapping,
        "rounds": rounds,
        "df": df,
        "config_df": config_df,
        "matching_token_price": matching_token_price,
        "scores": scores,
        "score_at_50_percent": score_at_50_percent,
        "score_at_100_percent": score_at_100_percent,
        "sybilDefense": sybilDefense,
        "chain_id": chain_id,
        "filter_in_list": filter_in_list
    }

def process_filterout_csv(df, csv):
    """Process uploaded CSV file for wallet filtering."""
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
    return df

def display_round_settings(data):
    """Display the settings and statistics for the current round."""
    st.title(f" {data['rounds']['round_name'].values[0]}: Matching Results")
    st.header(f"⚙️ Round Settings")
    col1, col2 = st.columns(2)
    col1.write(f"**Chain:** {data['blockchain_mapping'][data['chain_id']]}")
    col1.write(f"**Matching Cap:** {data['rounds']['matching_cap_amount'].values[0]:.2f}%")
    col1.write(f"**Passport Defense Selected:** {data['sybilDefense']}")
    col1.write(f"**Number of Unique Voters:** {data['df']['voter'].nunique()}")
    col2.write(f"**Matching Available:** {data['rounds']['matching_funds_available'].values[0]:.2f} {data['config_df']['token_code'].iloc[0]}")
    col2.write(f"**Matching Token Price:** ${data['matching_token_price']:.2f}")
    col2.write(f"**Minimum Donation Threshold Amount:** ${data['rounds'].get('min_donation_threshold_amount', 0).values[0]:.2f}")
    col2.write(f"**Number of Unique Projects:** {data['df']['project_name'].nunique()}")


def calculate_percent_scored_voters(data):
    """Calculate the percentage of unique voters who have a score."""
    total_unique_voters = data['df']['voter'].nunique()
    scored_unique_voters = len(data['scores'])
    percent_scored = (scored_unique_voters / total_unique_voters) * 100 if total_unique_voters > 0 else 0
    return percent_scored

def display_scores_progress_bar(data):
    """Display a progress bar showing the percentage of voters who have a score."""
    percent_scored = calculate_percent_scored_voters(data)
    st.subheader('')
    st.subheader(f"{percent_scored:.2f}% of addresses scored using {data['sybilDefense']}")
    st.progress(percent_scored/100)
    st.subheader('')

def display_crowdfunding_stats(df, matching_amount_display, matching_amount):
    """Display crowdfunding statistics and metrics."""
    st.header('👥 Crowdfunding')
    crowd_raised = df['amountUSD'].sum()
    st.subheader(f"${crowd_raised:,.2f} raised by crowd")
    st.subheader(f"{(crowd_raised / matching_amount_display) * 100:.2f}% of the matching pool")
    
    grouped_voter_data = df.groupby('voter')['amountUSD'].agg(['sum', 'mean', 'median', 'max']).reset_index()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Median Donor Contribution", f"${grouped_voter_data['median'].median():.2f}")
    col2.metric("Average Donor Contribution", f"${grouped_voter_data['mean'].mean():.2f}")
    col3.metric("Max Donor Contribution", f"${grouped_voter_data['max'].max():.2f}")

    return grouped_voter_data

def create_donation_distribution_chart(grouped_voter_data):
    """Create a chart showing the distribution of donor contributions."""
    bin_edges = [0, 1, 2, 3, 4, 5, 10, 20, 30, 50, 100, 500, 1000, np.inf]
    bin_labels = ['0-1','1-2','2-3','3-4', '4-5', '5-10', '10-20','20-30','30-50', '50-100', '100-500', '500-1000', '1000+']
    grouped_voter_data['amountUSD_bin'] = pd.cut(grouped_voter_data['sum'], bins=bin_edges, labels=bin_labels, right=False)
    
    fig = px.histogram(grouped_voter_data, x="amountUSD_bin", category_orders={'amountUSD_bin': bin_labels},
                       labels={'amountUSD_bin': 'Donation Amount Range (USD)'}, nbins=len(bin_edges)-1)
    fig.update_traces(hovertemplate='<b>Donation Range:</b> $%{x}<br><b>Number of Donors:</b> %{y}')
    fig.update_layout(
        title_text='Distribution of Donor Contributions by Amount',
        xaxis=dict(title='Donation Amount Range (USD)', titlefont=dict(size=18), tickfont=dict(size=14)),
        yaxis=dict(title='Number of Donors', titlefont=dict(size=18), tickfont=dict(size=14)),
        bargap=0.3
    )
    return fig

def handle_csv_upload(purpose='filter out'):
    """Handle the upload and processing of CSV file for wallet filtering."""
    if purpose == 'filter out':
        st.write('Upload a CSV file with a single column named "address" containing the ETH addresses to filter out. Addresses should include the 0x prefix.')
    if purpose == 'filter in':
        st.write('Upload a CSV file with a single column named "address" containing the ETH addresses to filter in. Addresses should include the 0x prefix. These addresses will be exempt from passport-based sybil detection.')
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv", key=purpose)
    if uploaded_file is not None:
        csv = pd.read_csv(uploaded_file)
        st.write("CSV file uploaded successfully. Here's a preview:")
        st.write(csv.head())
        return csv
    return None

def calculate_verified_vs_unverified(scores, donations_df, score_threshold):
    """Calculate and visualize the comparison between verified and unverified users."""
    verified_mask = scores['rawScore'] >= score_threshold
    verified_users_count = verified_mask.sum()
    verified_funding_total = donations_df.loc[donations_df['rawScore'] >= score_threshold, 'amountUSD'].sum()
    unverified_users_count = (~verified_mask).sum()
    unverified_funding_total = donations_df.loc[donations_df['rawScore'] < score_threshold, 'amountUSD'].sum()

    summary_data = pd.DataFrame({
        'Category': ['Verified', 'Unverified'],
        'Users': [verified_users_count, unverified_users_count],
        'Crowdfunding': [verified_funding_total, unverified_funding_total]
    })

    summary_data['Percentage Of Users'] = summary_data['Users'] / summary_data['Users'].sum() * 100
    summary_data['Percentage Of Crowdfunding'] = summary_data['Crowdfunding'] / summary_data['Crowdfunding'].sum() * 100

    fig = go.Figure()
    for metric in ['Crowdfunding', 'Users']:
        fig.add_trace(go.Bar(
            x=summary_data[f'Percentage Of {metric}'],
            y=summary_data['Category'],
            orientation='h',
            name=f'Percentage Of {metric}',
            text=summary_data[f'Percentage Of {metric}'].apply(lambda x: f"{x:.1f}%"),
            textposition='auto',
            hovertemplate=f'<b>%{{y}}</b><br>Percentage Of {metric}: %{{x:.1f}}%<br>{metric}: %{{customdata:,}}<extra></extra>',
            customdata=summary_data[metric]
        ))

    fig.update_layout(
        title_text='Percentage of Crowdfunding and Users by Passport Status',
        title_font=dict(size=24),
        bargap=0.3,
        xaxis=dict(title='Percentage', titlefont=dict(size=18), tickfont=dict(size=14)),
        yaxis=dict(title='Category', titlefont=dict(size=18), tickfont=dict(size=14)),
        barmode='group',
        legend=dict(traceorder='reversed'),
        hoverlabel=dict(bgcolor="white", font_size=16, font_family="Rockwell")
    )
    return fig, summary_data

def display_passport_usage(data):
    """Display passport usage statistics if Sybil defense is enabled."""
    if data['sybilDefense'] != 'None':
        st.header('🛂 Passport Usage')
        display_scores_progress_bar(data)
        num_filtered_in = len(data['filter_in_list'])
        total_voters = data['df']['voter'].nunique()
        if data['sybilDefense'] in ['Passport Stamps', 'Avalanche Passport']:
            st.subheader(f" {len(data['scores'])} Users ({len(data['scores'])/total_voters*100:.1f}%) Have a Passport Score")
            if num_filtered_in > 0:
                st.write(f'{num_filtered_in} users manually filtered in')
            passport_usage_fig, passport_usage_df = calculate_verified_vs_unverified(data['scores'], data['df'], data['score_at_50_percent'])
            st.plotly_chart(passport_usage_fig, use_container_width=True)
        if data['sybilDefense'] == 'Passport Model Based Detection System':
            total_voters = data['df']['voter'].nunique()
            n_users_passing_100 = len(data['scores'][data['scores']['rawScore'] >= data['score_at_100_percent']]) 
            n_users_passing_50 = len(data['scores'][(data['scores']['rawScore'] >= data['score_at_50_percent']) & (data['scores']['rawScore'] < data['score_at_100_percent'])])
            st.subheader(f" {n_users_passing_100} Users ({n_users_passing_100/total_voters*100:.1f}%) recieve full matching (passport model score over {data['score_at_100_percent']})")
            st.subheader(f" {n_users_passing_50} Users ({n_users_passing_50/total_voters*100:.1f}%) recieve partial matching (passport model score between {data['score_at_50_percent']} and {data['score_at_100_percent']})")
            if num_filtered_in > 0:
                st.write(f'{num_filtered_in} users manually filtered in')

def calculate_matching_results(data):
    """Calculate matching results using different strategies (COCM and QF)."""
    # Apply voting eligibility based on passport scores and donation thresholds
    df_with_passport = fundingutils.apply_voting_eligibility(
        data['df'].copy(), 
        data['rounds'].get('min_donation_threshold_amount', 0).values[0],
        data['score_at_50_percent'],
        data['score_at_100_percent'],
        data['filter_in_list']
    )
    votes_df_with_passport = fundingutils.pivot_votes(df_with_passport)
    
    matching_cap_amount = data['rounds']['matching_cap_amount'].astype(float).values[0]
    matching_amount = data['rounds']['matching_funds_available'].astype(float).values[0]
    
    # right now data['strat'] is either "half-and-half" or "COCM"
    s = data['strat']

    # Calculate matching amounts using both COCM and QF strategies
    matching_dfs = [fundingutils.get_qf_matching(strategy, votes_df_with_passport, matching_cap_amount, matching_amount, cluster_df=votes_df_with_passport) 
                    for strategy in [s, 'QF']]
    
    # Merge results from both strategies
    matching_df = pd.merge(matching_dfs[0], matching_dfs[1], on='project_name', suffixes=(f'_{s}', '_QF'))
    
    # Add project details and calculate the difference between COCM and QF matching
    df_unique = data['df'][['project_name', 'chain_id', 'round_id', 'application_id']].drop_duplicates()
    matching_df = pd.merge(matching_df, df_unique, on='project_name', how='left')
    matching_df['Project Page'] = 'https://explorer.gitcoin.co/#/round/' + matching_df['chain_id'].astype(str) + '/' + matching_df['round_id'].astype(str) + '/' + matching_df['application_id'].astype(str)
    matching_df['Δ Match'] = matching_df[f'matching_amount_{s}'] - matching_df['matching_amount_QF']
    
    return matching_df.sort_values(f'matching_amount_{s}', ascending=False)

def display_matching_results(matching_df, matching_token_symbol, s):
    """Display the matching results in a formatted table."""
    st.subheader('Matching Results')
    column_config = {
        "project_name": st.column_config.TextColumn("Project"),
        f"matching_amount_{s}": st.column_config.NumberColumn(f"{s} Match", format="%.2f"),
        "matching_amount_QF": st.column_config.NumberColumn("QF Match", format="%.2f"),
        "Δ Match": st.column_config.NumberColumn("Δ Match", format="%.2f"),
        "Project Page": st.column_config.LinkColumn("Project Page", display_text="Visit")
    }
    
    display_columns = ['project_name', f'matching_amount_{s}', 'matching_amount_QF', 'Δ Match', 'Project Page']
    st.dataframe(
        matching_df[display_columns],
        use_container_width=True,
        column_config=column_config,
        hide_index=True
    )
    
    st.markdown(f'Matching Values shown above are in **{matching_token_symbol}**')

def select_matching_strategy(s):
    """Allow user to select the matching strategy for download."""
    return st.selectbox(
        'Select the matching strategy to download:',
        (f'{s}', 'QF')
    )

def prepare_output_dataframe(matching_df, strategy_choice, data):
    """Prepare the output dataframe for download based on the selected strategy."""
    # Select relevant columns and rename them
    output_df = matching_df[['project_name', f'matching_amount_{strategy_choice}']]
    output_df = output_df.rename(columns={f'matching_amount_{strategy_choice}': 'matched'})
    
    # Fetch and merge project details
    projects_df = utils.get_projects_in_round(data['rounds']['round_id'].iloc[0], data['chain_id'])
    projects_df = projects_df[~projects_df['project_name'].isin(data['projects_to_remove'])]
    
    output_df = pd.merge(output_df, projects_df, left_on='project_name', right_on='project_name', how='outer')
    output_df = output_df.rename(columns={
        'id': 'applicationId', 
        'project_id': 'projectId', 
        'project_name': 'projectName', 
        'recipient_address': 'payoutAddress', 
        'total_donations_count': 'contributionsCount',
        'total_amount_donated_in_usd': 'totalReceived'
    })
    
    # Convert matching amounts to USD and adjust for token decimals
    matching_token_decimals = data['config_df']['token_decimals'].iloc[0]
    output_df['matchedUSD'] = (output_df['matched'] * data['matching_token_price']).round(2)
    
    output_df['matched'] = (output_df['matched'] * 10**matching_token_decimals).apply(lambda x: int(x))
    output_df['totalReceived'] = (output_df['totalReceived'] * 10**matching_token_decimals).apply(lambda x: int(x))
    
    # Reorder columns and add placeholder columns required for the output format
    output_df = output_df[[
        'applicationId', 'projectId', 'projectName', 'payoutAddress', 
        'matchedUSD', 'totalReceived', 'contributionsCount', 'matched'
    ]]
    
    output_df['sumOfSqrt'] = 0
    output_df['capOverflow'] = 0
    output_df['matchedWithoutCap'] = 0
    
    return output_df.fillna(0)

def adjust_matching_overflow(output_df, matching_funds_available, matching_token_decimals):
    """Adjust matching funds if there's an overflow."""
    full_matching_funds_available = int(int(matching_funds_available) * 10**(int(matching_token_decimals)))
    matching_overflow = sum(int(x) for x in output_df['matched']) - full_matching_funds_available
    while matching_overflow >= 0:
        st.warning(f'Potential Matching Overflow of {matching_overflow} Detected. Adjusting Matching Funds')
        output_df['matched_pct'] = output_df['matched'] / output_df['matched'].sum()
        output_df['matched'] = (output_df['matched'] - (output_df['matched_pct'] * matching_overflow).apply(lambda x: int(x))).clip(lower=1)
        matching_overflow = sum(int(x) for x in output_df['matched']) - full_matching_funds_available
        st.warning(f'Adjusted Matching Overflow is {matching_overflow}') # IF THIS NUMBER IS NEGATIVE WE ARE GOOD TO GO
    output_df['matched'] = output_df['matched'].apply(lambda x: int(x))
    #output_df = output_df.drop(columns=['matched_pct'])
    return output_df

def display_matching_distribution(output_df):
    """Display and provide download option for the matching distribution."""
    # Create a copy for display purposes
    display_df = output_df.copy()
    
    # Format large numbers for display
    for col in ['matched', 'totalReceived', 'sumOfSqrt', 'capOverflow', 'matchedWithoutCap']:
        display_df[col] = display_df[col].apply(lambda x: f"{x}")
    
    st.write(display_df)
    
    # Use the original output_df for CSV download
    st.download_button(
        label="⬇ Download Matching Distribution",
        data=output_df.to_csv(index=False),
        file_name='matching_distribution.csv',
        mime='text/csv'
    )
    st.write('You can upload this CSV to manager.gitcoin.co to apply the matching results to your round')
    #st.header(f'The value of the sum of the matched column is {output_df["matched"].sum()}')

def create_summary_dataframe(output_df, matching_df, token_code, s):
    """Create a summary dataframe for the round results."""
    summary_df = output_df[['projectName', 'matchedUSD']].copy()
    summary_df = summary_df.merge(matching_df[['project_name', f'matching_amount_{s}', 'Project Page']], 
                                  left_on='projectName', right_on='project_name', how='left')
    summary_df = summary_df.rename(columns={
        'projectName': 'Project',
        f'matching_amount_{s}': f'Matching Funds ({token_code})',
        'matchedUSD': 'Matching Funds (USD)',
        'Project Page': 'Project Page'
    })
    summary_df = summary_df[[
        'Project', 
        f'Matching Funds ({token_code})', 
        'Matching Funds (USD)', 
        'Project Page'
    ]]
    
    for col in ['Matching Funds (USD)', f'Matching Funds ({token_code})']:
        summary_df[col] = summary_df[col].round(2)
    
    return summary_df.sort_values(f'Matching Funds ({token_code})', ascending=False)

def display_summary(summary_df):
    """Display and provide download option for the round summary."""
    st.write(summary_df)
    st.download_button(
        label="⬇ Download Summary",
        data=summary_df.to_csv(index=False),
        file_name='round_summary.csv',
        mime='text/csv'
    )

def create_matching_distribution_chart(summary_df, token_symbol):
    """Create a bar chart showing the distribution of matching funds."""
    summary_df = summary_df.sort_values(f'Matching Funds ({token_symbol})', ascending=True)
    fig = px.bar(
        summary_df,
        x=f'Matching Funds ({token_symbol})',
        y='Project',
        orientation='h',
        title='Matching Funds Distribution',
        labels={f'Matching Funds ({token_symbol})': 'Matched Funds', 'Project': 'Project'},
        text=summary_df[f'Matching Funds ({token_symbol})'].apply(lambda x: f"{x/1000:.1f}k" if x >= 1000 else f"{x:.0f}")
    )
    fig.update_layout(
        xaxis_title='',
        yaxis_title='',
        yaxis=dict(tickmode='linear'),
        template='plotly_white',
        height=1640,
        width=800
    )
    return fig

def main():
    """Main function to run the Streamlit app."""
    st.image('assets/657c7ed16b14af693c08b92d_GTC-Logotype-Dark.png', width=200)
    round_id, chain_id = validate_input()
    
    # Advanced options 
    with st.expander("Advanced: Filter Out Wallets", expanded=False):
        filter_out_csv = handle_csv_upload(purpose='filter out')

    with st.expander("Advanced: Filter In Wallets", expanded=False):
        filter_in_csv = handle_csv_upload(purpose='filter in')

    half_and_half = False
    with st.expander("Advanced: Give results as half COCM / half QF"):
        st.write('''
            Toggle the switch below to calculate a matching result blending COCM and QF, instead of pure COCM.
            In this case, funding amounts will be found for each mechanism, normalized to the size of the matching pool, and then averaged for each project.
            E.g. if a project gets 10% of the matching pool under COCM and 40% of the matching pool under QF, they will get 25% of the matching pool under this calculation.''')
        half_and_half = st.toggle('Select for half-and-half')

    # Load and process data
    data = load_data(round_id, chain_id, filter_out_csv=filter_out_csv, filter_in_csv=filter_in_csv)

    if half_and_half == True:
        data['strat'] = 'half-and-half'
    else:
        data['strat'] = 'COCM'

    # Display various settings of the round
    display_round_settings(data)

    matching_amount = data['rounds']['matching_funds_available'].astype(float).values[0]
    matching_amount_display = data['matching_token_price'] * matching_amount

    # Crowdfunding stats section
    grouped_voter_data = display_crowdfunding_stats(data['df'], matching_amount_display, matching_amount)
    st.plotly_chart(create_donation_distribution_chart(grouped_voter_data), use_container_width=True)

    # Passport usage section (ONLY APPEARS IF SYBIL DEFENSE IS ENABLED)
    display_passport_usage(data)

    # Quadratic Funding Method Comparison section
    st.header('💚 Quadratic Funding Method Comparison')
    st.write('''[Quadratic funding](https://wtfisqf.com) helps us solve coordination failures by creating a way for community members to fund what matters to them while amplifying their impact. However, its assumption that people make independent decisions can be exploited to unfairly influence the distribution of matching funds.''')
    st.write('''[Connection-oriented cluster-matching (COCM)](https://wtfiscocm.streamlit.app/) doesn't make this assumption. Instead, it quantifies just how coordinated groups of actors are likely to be based on the social signals they have in common. Projects backed by more independent agents receive greater matching funds. Conversely, if a project's support network shows higher levels of coordination, the matching funds are reduced, encouraging self-organized solutions within more coordinated groups. ''')
    
    # Allow removal of projects from matching distribution
    all_projects = data['df']['project_name'].unique()
    data['projects_to_remove'] = st.multiselect('Projects may be removed from the matching distribution by selecting them here:', all_projects)
    data['df'] = data['df'][~data['df']['project_name'].isin(data['projects_to_remove'])]
    
    # Calculate and display matching results
    matching_df = calculate_matching_results(data)
    display_matching_results(matching_df, data['config_df']['token_code'].iloc[0], data['strat'])

    # Matching distribution download section
    st.subheader('Download Matching Distribution')
    if calculate_percent_scored_voters(data) < 100:
        st.warning('Matching distribution download is not recommended until 100% of addresses are scored. This could take 24-72 hours after the round concludes.')
    strategy_choice = select_matching_strategy(data['strat'])
    output_df = prepare_output_dataframe(matching_df, strategy_choice, data)
    output_df = adjust_matching_overflow(output_df, data['rounds']['matching_funds_available'].values[0], data['config_df']['token_decimals'].iloc[0])
    display_matching_distribution(output_df)

    # Summary section
    st.header('📈 Sharable Summary')
    token_code = data['config_df']['token_code'].iloc[0]
    summary_df = create_summary_dataframe(output_df, matching_df, token_code, data['strat'])
    display_summary(summary_df)

    #matching_distribution_chart = create_matching_distribution_chart(summary_df, token_code)
    #st.plotly_chart(matching_distribution_chart)

if __name__ == "__main__":
    main()