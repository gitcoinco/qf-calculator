import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
import utils
import fundingutils

# Page configuration
st.set_page_config(page_title="Matching Results", page_icon="favicon.png", layout="wide")

# Session state initialization
if 'round_id' not in st.session_state:
    st.session_state.round_id = None
if 'chain_id' not in st.session_state:
    st.session_state.chain_id = None

# URL parameter handling
query_params_round_id = st.query_params.get_all('round_id')
if len(query_params_round_id) == 1 and not st.session_state.round_id:
    st.session_state.round_id = query_params_round_id[0]

query_params_chain_id = st.query_params.get_all('chain_id')
if len(query_params_chain_id) == 1 and not st.session_state.chain_id:
    st.session_state.chain_id = query_params_chain_id[0]

def validate_input():
    if st.session_state.round_id is None or st.session_state.chain_id is None:
        st.header("Oops! Something went wrong. You're not supposed to be here 🙈")
        st.subheader("Please provide round_id and chain_id in the URL")
        st.subheader('Example: https://qf-calculator.fly.dev/?round_id=23&chain_id=42161')
        st.stop()
    return st.session_state.round_id.lower(), int(st.session_state.chain_id)

@st.cache_resource(ttl=36000)
def load_scores_and_set_defense(chain_id, sybilDefense, unique_voters):
    if chain_id == 43114: 
        scores = utils.load_avax_scores(unique_voters)
        score_at_50_percent = score_at_100_percent = 25
        sybilDefense = 'Avalanche Passport'
    elif sybilDefense == 'true':
        scores = utils.load_stamp_scores(unique_voters)
        score_at_50_percent, score_at_100_percent = 15, 25
        sybilDefense = 'Passport Stamps'
    elif sybilDefense == 'passport-mbds':
        scores = utils.load_passport_model_scores(unique_voters)
        score_at_50_percent, score_at_100_percent = 1, 25
        sybilDefense = 'Passport Model Based Detection System'
    else:
        scores = pd.DataFrame({'address': unique_voters, 'rawScore': 1})
        score_at_50_percent = score_at_100_percent = 0
        sybilDefense = 'None'
    return scores, score_at_50_percent, score_at_100_percent, sybilDefense

def load_data(round_id, chain_id, csv=None):
    blockchain_mapping = {1: "Ethereum", 10: "Optimism", 137: "Polygon", 250: "Fantom",
                          324: "ZKSync", 8453: "Base", 42161: "Arbitrum", 43114: "Avalanche",
                          534352: "Scroll", 1329: "SEI"}
    rounds = utils.get_round_summary()
    rounds = rounds[(rounds['round_id'].str.lower() == round_id) & (rounds['chain_id'] == chain_id)]
    token = rounds['token'].values[0] if 'token' in rounds else 'ETH'
    sybilDefense = rounds['sybil_defense'].values[0] if 'sybil_defense' in rounds else 'None'
    df = utils.get_round_votes(round_id, chain_id)
    
    if csv is not None:
        df = process_csv(df, csv)
    
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
        "chain_id": chain_id  # Include chain_id in the returned data
    }

def process_csv(df, csv):
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
    st.header('⚙️ Round Settings')
    col1, col2 = st.columns(2)
    col1.write(f"**Chain:** {data['blockchain_mapping'][data['chain_id']]}")
    col1.write(f"**Matching Cap:** {data['rounds']['matching_cap_amount'].values[0]:.2f}%")
    col1.write(f"**Passport Defense Selected:** {data['sybilDefense']}")
    col1.write(f"**Number of Unique Voters:** {data['df']['voter'].nunique()}")
    col2.write(f"**Matching Available:** {data['rounds']['matching_funds_available'].values[0]:.2f} {data['config_df']['token_code'].iloc[0]}")
    col2.write(f"**Matching Token Price:** ${data['matching_token_price']:.2f}")
    col2.write(f"**Minimum Donation Threshold Amount:** ${data['rounds'].get('min_donation_threshold_amount', 0).values[0]:.2f}")
    col2.write(f"**Number of Unique Projects:** {data['df']['project_name'].nunique()}")

def display_crowdfunding_stats(df, matching_amount_display, matching_amount):
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

def handle_wallet_filtering():
    st.write('Upload a CSV file with a single column named "address" containing the ETH addresses to filter out. Addresses should include the 0x prefix.')
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
    if uploaded_file is not None:
        csv = pd.read_csv(uploaded_file)
        st.write("CSV file uploaded successfully. Here's a preview:")
        st.write(csv.head())
        return csv
    return None

def handle_matching_funds_override(rounds):
    matching_funds_available = st.number_input("Matching Funds Available", 
                                               value=rounds['matching_funds_available'].astype(float).values[0], 
                                               format="%.2f")
    rounds['matching_funds_available'] = matching_funds_available
    return rounds

def calculate_verified_vs_unverified(scores, donations_df, score_threshold):
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
    if data['sybilDefense'] != 'None':
        st.header('🛂 Passport Usage')
        total_voters = data['df']['voter'].nunique()
        st.subheader(f" {len(data['scores'])} Users ({len(data['scores'])/total_voters*100:.1f}%) Have a Passport Score")
        passport_usage_fig, passport_usage_df = calculate_verified_vs_unverified(data['scores'], data['df'], data['score_at_50_percent'])
        st.plotly_chart(passport_usage_fig, use_container_width=True)

def calculate_matching_results(data):
    df_with_passport = fundingutils.apply_voting_eligibility(
        data['df'].copy(), 
        data['rounds'].get('min_donation_threshold_amount', 0).values[0],
        data['score_at_50_percent'],
        data['score_at_100_percent']
    )
    votes_df_with_passport = fundingutils.pivot_votes(df_with_passport)
    
    matching_cap_amount = data['rounds']['matching_cap_amount'].astype(float).values[0]
    matching_amount = data['rounds']['matching_funds_available'].astype(float).values[0]
    
    matching_dfs = [fundingutils.get_qf_matching(strategy, votes_df_with_passport, matching_cap_amount, matching_amount, cluster_df=votes_df_with_passport) 
                    for strategy in ['COCM', 'QF']]
    
    matching_df = pd.merge(matching_dfs[0], matching_dfs[1], on='project_name', suffixes=('_COCM', '_QF'))
    
    df_unique = data['df'][['project_name', 'chain_id', 'round_id', 'application_id']].drop_duplicates()
    matching_df = pd.merge(matching_df, df_unique, on='project_name', how='left')
    matching_df['Project Page'] = 'https://explorer.gitcoin.co/#/round/' + matching_df['chain_id'].astype(str) + '/' + matching_df['round_id'].astype(str) + '/' + matching_df['application_id'].astype(str)
    matching_df['Δ Match'] = matching_df['matching_amount_COCM'] - matching_df['matching_amount_QF']
    
    return matching_df.sort_values('matching_amount_COCM', ascending=False)
    
    return matching_df.sort_values('matching_amount_COCM', ascending=False)

def display_matching_results(matching_df, matching_token_symbol):
    st.subheader('Matching Results')
    column_config = {
        "project_name": st.column_config.TextColumn("Project"),
        "matching_amount_COCM": st.column_config.NumberColumn("COCM Match", format="%.2f"),
        "matching_amount_QF": st.column_config.NumberColumn("QF Match", format="%.2f"),
        "Δ Match": st.column_config.NumberColumn("Δ Match", format="%.2f"),
        "Project Page": st.column_config.LinkColumn("Project Page", display_text="Visit")
    }
    
    display_columns = ['project_name', 'matching_amount_COCM', 'matching_amount_QF', 'Δ Match', 'Project Page']
    st.dataframe(
        matching_df[display_columns],
        use_container_width=True,
        column_config=column_config,
        hide_index=True
    )
    
    st.markdown(f'Matching Values shown above are in **{matching_token_symbol}**')

def select_matching_strategy():
    return st.selectbox(
        'Select the matching strategy to download:',
        ('COCM', 'QF')
    )

def prepare_output_dataframe(matching_df, strategy_choice, data):
    output_df = matching_df[['project_name', f'matching_amount_{strategy_choice}']]
    output_df = output_df.rename(columns={f'matching_amount_{strategy_choice}': 'matched'})
    
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
    
    matching_token_decimals = data['config_df']['token_decimals'].iloc[0]
    output_df['matchedUSD'] = (output_df['matched'] * data['matching_token_price']).round(2)
    
    output_df['matched'] = (output_df['matched'] * 10**matching_token_decimals).apply(lambda x: int(x))
    output_df['totalReceived'] = (output_df['totalReceived'] * 10**matching_token_decimals).apply(lambda x: int(x))
    
    output_df = output_df[[
        'applicationId', 'projectId', 'projectName', 'payoutAddress', 
        'matchedUSD', 'totalReceived', 'contributionsCount', 'matched'
    ]]
    
    output_df['sumOfSqrt'] = 0
    output_df['capOverflow'] = 0
    output_df['matchedWithoutCap'] = 0
    
    return output_df.fillna(0)

def adjust_matching_overflow(output_df, matching_funds_available, matching_token_decimals):
    full_matching_funds_available = int(matching_funds_available * 10**matching_token_decimals)
    matching_overflow = sum(int(x) for x in output_df['matched']) - full_matching_funds_available
    
    if matching_overflow >= 0:
        st.warning('Potential Matching Overflow Detected. Adjusting Matching Funds')
        matching_adjustment = int(matching_overflow / max(output_df['matched'].count(), 1))
        output_df['matched'] = output_df['matched'].apply(lambda x: str(max(int(x) - matching_adjustment, 0)))
        matching_overflow = sum(int(x) for x in output_df['matched']) - full_matching_funds_available
        st.warning(f'Adjusted Matching Overflow is {matching_overflow}')
    
    
    return output_df

def display_matching_distribution(output_df):
    # Create a copy for display purposes
    display_df = output_df.copy()

    
    # Format large numbers for display
    for col in ['matched', 'totalReceived', 'sumOfSqrt', 'capOverflow', 'matchedWithoutCap']:
        display_df[col] = display_df[col].apply(lambda x: f"{x:,}")
    
    st.write(display_df)
    
    # Use the original output_df for CSV download
    st.download_button(
        label="⬇ Download Matching Distribution",
        data=output_df.to_csv(index=False),
        file_name='matching_distribution.csv',
        mime='text/csv'
    )
    st.write('You can upload this CSV to manager.gitcoin.co to apply the matching results to your round')

def create_summary_dataframe(output_df, matching_df, token_code):
    summary_df = output_df[['projectName', 'matchedUSD']].copy()
    summary_df = summary_df.merge(matching_df[['project_name', 'matching_amount_COCM', 'Project Page']], 
                                  left_on='projectName', right_on='project_name', how='left')
    summary_df = summary_df.rename(columns={
        'projectName': 'Project',
        'matching_amount_COCM': f'Matching Funds ({token_code})',
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
    st.write(summary_df)
    st.download_button(
        label="⬇ Download Summary",
        data=summary_df.to_csv(index=False),
        file_name='round_summary.csv',
        mime='text/csv'
    )

def create_matching_distribution_chart(summary_df, token_symbol):
    fig = px.bar(
        summary_df.sort_values(f'Matching Funds ({token_symbol})', ascending=True),
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
    st.image('657c7ed16b14af693c08b92d_GTC-Logotype-Dark.png', width=300)
    
    round_id, chain_id = validate_input()
    
    with st.expander("Advanced: Filter Out Wallets", expanded=False):
        csv = handle_wallet_filtering()

    data = load_data(round_id, chain_id, csv)

    with st.expander("Advanced: Override Matching Funds Available", expanded=False):
        data['rounds'] = handle_matching_funds_override(data['rounds'])

    display_round_settings(data)

    matching_amount = data['rounds']['matching_funds_available'].astype(float).values[0]
    matching_amount_display = data['matching_token_price'] * matching_amount

    grouped_voter_data = display_crowdfunding_stats(data['df'], matching_amount_display, matching_amount)

    st.plotly_chart(create_donation_distribution_chart(grouped_voter_data), use_container_width=True)

    display_passport_usage(data)

    st.header('💚 Quadratic Funding Method Comparison')
    st.write('''Quadratic funding helps us solve coordination failures by creating a way for community members to fund what matters to them while amplifying their impact. However, its assumption that people make independent decisions can be exploited to unfairly influence the distribution of matching funds.''')
    st.write('''Connection-oriented cluster-matching (COCM) doesn't make this assumption. Instead, it quantifies just how coordinated groups of actors are likely to be based on the social signals they have in common. Projects backed by more independent agents receive greater matching funds. Conversely, if a project's support network shows higher levels of coordination, the matching funds are reduced, encouraging self-organized solutions within more coordinated groups. ''')

    all_projects = data['df']['project_name'].unique()
    data['projects_to_remove'] = st.multiselect('Projects may be removed from the matching distribution by selecting them here:', all_projects)
    data['df'] = data['df'][~data['df']['project_name'].isin(data['projects_to_remove'])]

    matching_df = calculate_matching_results(data)
    display_matching_results(matching_df, data['config_df']['token_code'].iloc[0])

    st.subheader('Download Matching Distribution')
    strategy_choice = select_matching_strategy()
    output_df = prepare_output_dataframe(matching_df, strategy_choice, data)
    output_df = adjust_matching_overflow(output_df, data['rounds']['matching_funds_available'].values[0], data['config_df']['token_decimals'].iloc[0])
    
    display_matching_distribution(output_df)

    st.header('📈 Sharable Summary')
    token_code = data['config_df']['token_code'].iloc[0]
    summary_df = create_summary_dataframe(output_df, matching_df, token_code)
    display_summary(summary_df)

    matching_distribution_chart = create_matching_distribution_chart(summary_df, token_code)
    st.plotly_chart(matching_distribution_chart)

if __name__ == "__main__":
    main()