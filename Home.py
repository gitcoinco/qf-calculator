from queries.graphql.round_summary import get_round_summary_graphql
from queries.graphql.recent_rounds import get_recent_rounds_graphql
from queries.graphql.project_summary import get_project_summary_graphql
from queries.graphql.votes_by_round import get_votes_by_round_graphql
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
import networkx as nx
import utils
import fundingutils
from decimal import Decimal, getcontext
import decimal


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
    rounds = get_recent_rounds_graphql(limit=100)

    # Create round links and prepare display data
    rounds['Round Link'] = rounds.apply(lambda row: f"https://qf-calculator.fly.dev/?round_id={row['round_id']}&chain_id={row['chain_id']}", axis=1)
    rounds_display = rounds[[
        'round_name', 
        'chain_id',
        'round_id',
        'Round Link',
        'votes',
        'uniqueContributors',
        'amountUSD'
    ]]
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
    st.header("Recent Rounds That Have Ended With More than 10 Unique Contributors:")
    st.dataframe(
        rounds_display,
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
    if chain_id == 43114 and sybilDefense != 'none':  # AVALANCHE 
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

def check_round_existence(round_id, chain_id):
    rounds = get_round_summary_graphql(chain_id, round_id)
    rounds = rounds[(rounds['round_id'].str.lower() == round_id) & (rounds['chain_id'] == chain_id)] # FILTER BY ROUND_ID AND CHAIN_ID
    if len(rounds) == 0:
        st.write('## We could not find your round in our data.')
        st.write('')
        display_recent_rounds()
        st.stop()

def load_data(round_id, chain_id):
    """Load and process data for the specified round and chain."""
    blockchain_mapping = {1: "Ethereum", 10: "Optimism", 137: "Polygon", 250: "Fantom",
                          324: "ZKSync", 8453: "Base", 42161: "Arbitrum", 43114: "Avalanche",
                          534352: "Scroll", 1329: "SEI", 42220: "Celo", 1088: "Metis", 42: "Lukso" }
    rounds = get_round_summary_graphql(chain_id, round_id)
    
    token = rounds['token'].values[0] if 'token' in rounds else 'ETH'    
    sybilDefense = rounds['sybilDefense'].values[0] if 'sybilDefense' in rounds else 'None'
    df = get_votes_by_round_graphql(chain_id, round_id)

    with open("votes.txt", "w") as f:
        f.write(df.to_string())
        
    # Fetch token configuration and price
    config_df = utils.fetch_tokens_config()
    config_df = config_df[(config_df['chain_id'] == chain_id) & (config_df['token_address'] == token.lower())]
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
        "matching_cap": float(rounds['matching_cap_amount'].values[0]) if rounds['matching_cap_amount'].values[0] is not None else 0.0,
        "matching_available": float(rounds['matching_funds_available'].values[0]) if rounds['matching_funds_available'].values[0] is not None else 0.0,

    }

def display_round_settings(data):
    """Display the settings and statistics for the current round."""
    st.title(f" {data['rounds']['round_name'].values[0]}: Matching Results")
    st.header(f"⚙️ Round Settings")
    col1, col2 = st.columns(2)
    col1.write(f"**Chain:** {data['blockchain_mapping'][data['chain_id']]}")
    col1.write(f"**Matching Cap:** {data['matching_cap']:.2f}%")
    col1.write(f"**Passport Defense Selected:** {data['sybilDefense']}")
    col1.write(f"**Number of Unique Voters:** {data['df']['voter'].nunique()}")
    col1.write(f"**Total Donations Count:** {data['df']['voter'].count()}")
    col2.write(f"**Matching Available:** {data['matching_available']:.2f} {data['config_df']['token_code'].iloc[0]}")
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
        xaxis=dict(title='Donation Amount Range (USD)', title_font=dict(size=18), tickfont=dict(size=14)),
        yaxis=dict(title='Number of Donors', title_font=dict(size=18), tickfont=dict(size=14)),
        bargap=0.3
    )
    return fig

def handle_csv_upload(purpose='filter out'):
    """Handle the upload and processing of CSV file for wallet filtering."""
    if purpose == 'filter out':
        st.write('Upload a CSV file with a single column named "address" containing the ETH addresses to filter out. Addresses should include the 0x prefix.')
    if purpose == 'filter in':
        st.write('Upload a CSV file with a single column named "address" containing the ETH addresses to filter in. Addresses should include the 0x prefix. These addresses will be exempt from passport-based sybil detection.')
    if purpose == 'general scaling':
        st.write('Upload a CSV file with a column named "address" and a column named "scale". Addresses listed in the CSV will bypass passport scaling and instead have their contributions scaled by the amount listed. You do not need to include every address.')
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv", key=purpose)
    if uploaded_file is not None:
        csv = pd.read_csv(uploaded_file)

        st.subheader("✅ CSV file processed successfully and results updated below.")
        st.write("Preview of uploaded data:")
        st.dataframe(csv, height=200)
        csv['address'] = csv['address'].str.lower()

        csv.set_index('address', inplace=True)
        if purpose == 'filter in':
            csv['scale'] = 1
        if purpose == 'filter out':
            csv['scale'] = 0
        
        return csv
    return None

def display_network_graph(df):
    """Display a 3D network graph of voters and grants."""
    st.subheader('🌐 Connection Graph')
    grants_color = '#00433B'
    grantee_color_string = 'moss'
    voters_color = '#C4F092'
    voter_color_string = 'lightgreen'
    line_color = '#6E9A82'
    # Assuming 'data' is the DataFrame containing all donation information
    
    # Sum amountUSD grouped by voter and recipient_address
    grouped_data = df.groupby(['voter', 'recipient_address']).agg({
        'amountUSD': 'sum',
        'project_name': 'first'  # Assuming project_name is constant for each recipient_address
    }).reset_index()

    count_connections = grouped_data.shape[0]
    count_voters = grouped_data['voter'].nunique()
    count_grants = grouped_data['recipient_address'].nunique()
    max_connections = 2000
    col1, col2 = st.columns([3, 1])
        
    with col1:
        st.markdown("The below graph visualizes the connections between donors and grantees. " +
                    f"Donors are represented by {voter_color_string} nodes, while grantees are represented by {grantee_color_string} nodes. " +
                    f"Each line connecting a donor to a grantee represents a donation.")
        st.markdown("In COCM, projects receive higher matching when their donors support a diverse range of other projects and have unique connection patterns. Conversely, projects get lower matching if their donors primarily support a small number of the same projects.")

    with col2:
        pct_to_sample = st.slider("Percentage of connections to sample", 
                                  min_value=1, 
                                  max_value=100, 
                                  value=min(100, int(max_connections / count_connections * 100)),
                                  step=1,
                                  help="Adjust this to control the number of connections displayed in the graph. More connections means longer loading times")
        st.markdown("**Go fullscreen with the arrows in the top-right for a better view.**")

    num_connections_to_sample = int(count_connections * pct_to_sample / 100)
    grouped_data = grouped_data.sample(n=min(num_connections_to_sample, max_connections), random_state=42)
    count_connections = grouped_data.shape[0]


    # Initialize a new Graph
    B = nx.Graph()

    # Create nodes with the bipartite attribute
    B.add_nodes_from(grouped_data['voter'].unique(), bipartite=0, color=voters_color)
    B.add_nodes_from(grouped_data['recipient_address'].unique(), bipartite=1, color=grants_color)

    # Add edges with amountUSD as an attribute
    for _, row in grouped_data.iterrows():
        B.add_edge(row['voter'], row['recipient_address'], amountUSD=row['amountUSD'])

    pos = nx.spring_layout(B, dim=3, k=0.09, iterations=50)

    # Extract node information
    node_x, node_y, node_z = zip(*pos.values())
    node_names = list(pos.keys())
    degrees = [B.degree(node_name) for node_name in node_names]
    node_sizes = np.log(np.array(degrees) + 1) * 10

    # Extract edge information
    edge_x, edge_y, edge_z = [], [], []
    edge_weights = []

    for edge in B.edges(data=True):
        x0, y0, z0 = pos[edge[0]]
        x1, y1, z1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_z.extend([z0, z1, None])
        edge_weights.append(edge[2]['amountUSD'])

    # Create the edge traces
    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        line=dict(width=1, color=line_color),
        hoverinfo='none',
        mode='lines',
        marker=dict(opacity=0.5))

    # Create the node traces
    node_trace = go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            color=[data['color'] for _, data in B.nodes(data=True)],
            size=node_sizes,
            opacity=1,
            sizemode='diameter'
        ))

    node_adjacencies = [len(list(B.neighbors(node))) for node in node_names]
    node_trace.marker.color = [data['color'] for _, data in B.nodes(data=True)]

    # Prepare text information for hovering
    node_text = []
    for node in node_names:
        if node in grouped_data['recipient_address'].values:
            project_name = grouped_data[grouped_data['recipient_address'] == node]['project_name'].iloc[0]
            adj = len(list(B.neighbors(node)))
            connections_text = f"Connections: {adj}" if pct_to_sample == 100 else f"Sampled Connections: {adj}"
            node_text.append(f'Project: {project_name}<br>{connections_text}')
        else:
            adj = len(list(B.neighbors(node)))
            connections_text = f"Connections: {adj}" if pct_to_sample == 100 else f"Sampled Connections: {adj}"
            node_text.append(f'Voter: {node[:6]}...{node[-4:]}<br>{connections_text}')
    node_trace.text = node_text

    # Create the figure
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title=' ',
                        title_font=dict(size=20),
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        annotations=[dict(
                            showarrow=False,
                            text="Use your mouse to rotate, zoom, and pan around the 3D graph for a better view of connections.",
                            xref="paper",
                            yref="paper",
                            x=0.005,
                            y=-0.002)],
                        scene=dict(
                            xaxis_title='X Axis',
                            yaxis_title='Y Axis',
                            zaxis_title='Z Axis')))

    st.plotly_chart(fig, use_container_width=True)

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
        xaxis=dict(title='Percentage', title_font=dict(size=18), tickfont=dict(size=14)),
        yaxis=dict(title='Category', title_font=dict(size=18), tickfont=dict(size=14)),
        barmode='group',
        legend=dict(traceorder='reversed'),
        hoverlabel=dict(bgcolor="white", font_size=16, font_family="Rockwell")
    )
    return fig, summary_data

def categorize_user(score, score_at_50_percent, score_at_100_percent):
    if score >= score_at_100_percent:
        return 'Full'
    elif score >= score_at_50_percent:
        return 'Partial'
    else:
        return 'Unmatched'
                
def display_passport_usage(data):
    """Display passport usage statistics if Sybil defense is enabled."""
    if data['sybilDefense'] != 'None':
        st.header('🛂 Passport Usage')
        display_scores_progress_bar(data)
        num_adjusted = 0
        if data['scaling_df'] is not None:
            num_adjusted = len(data['scaling_df'])
        total_voters = data['df']['voter'].nunique()
        if data['sybilDefense'] in ['Passport Stamps', 'Avalanche Passport']:
            st.subheader(f" {len(data['scores'])} Users ({len(data['scores'])/total_voters*100:.1f}%) Have a Passport Score")
            if num_adjusted > 0:
                st.write(f'{num_adjusted} users had scores manually adjusted')
            passport_usage_fig, passport_usage_df = calculate_verified_vs_unverified(data['scores'], data['df'], data['score_at_50_percent'])
            st.plotly_chart(passport_usage_fig, use_container_width=True)
        if data['sybilDefense'] == 'Passport Model Based Detection System':
            total_voters = data['df']['voter'].nunique()
            n_users_passing_100 = len(data['scores'][data['scores']['rawScore'] >= data['score_at_100_percent']]) 
            n_users_passing_50 = len(data['scores'][(data['scores']['rawScore'] >= data['score_at_50_percent']) & (data['scores']['rawScore'] < data['score_at_100_percent'])])
            st.subheader(f" {n_users_passing_100} Users ({n_users_passing_100/total_voters*100:.1f}%) recieve full matching (passport model score over {data['score_at_100_percent']})")
            st.subheader(f" {n_users_passing_50} Users ({n_users_passing_50/total_voters*100:.1f}%) recieve partial matching (passport model score between {data['score_at_50_percent']} and {data['score_at_100_percent']})")
            st.markdown("""
                🔹 **Full matching:** User contributions are matched at 100% of the calculated amount.
                
                🔸 **Partial matching:** User contributions are matched between 50-100% of the calculated amount.
                
                📊 **Matching percentage is based on the user's passport model score:**
                - Higher scores = Higher matching percentage
                - Very low scores = No matching
                
                💡 This system encourages legitimate users to build stronger digital identities for better matching rates while protecting matching funds from sybils and airdrop farmers.
            """)


            with st.expander("**Expand: Matching Breakdown by Project Information**"):
                # Categorize donors based on their scores
                data['scores']['category'] = data['scores']['rawScore'].apply(
                    lambda score: categorize_user(score, data['score_at_50_percent'], data['score_at_100_percent'])
                )
                
                # Merge dataframes and fill missing categories
                merged_df = pd.merge(data['df'], data['scores'], left_on='voter', right_on='address', how='left')
                merged_df['category'] = merged_df['category'].fillna('Unmatched')
                
                # Group and pivot data
                grouped = merged_df.groupby(['project_name', 'category']).agg({
                    'voter': 'nunique',
                    'amountUSD': 'sum'
                }).reset_index()
                
                pivot_df = grouped.pivot(index='project_name', columns='category', values=['voter', 'amountUSD'])
                pivot_df.columns = [f'{col[1]}_{col[0]}' for col in pivot_df.columns]
                pivot_df = pivot_df.reset_index()
                
                # Ensure all necessary columns exist and replace None with 0
                for cat in ['Unmatched', 'Partial', 'Full']:
                    for val in ['voter', 'amountUSD']:
                        if f'{cat}_{val}' not in pivot_df.columns:
                            pivot_df[f'{cat}_{val}'] = 0
                        else:
                            pivot_df[f'{cat}_{val}'] = pivot_df[f'{cat}_{val}'].fillna(0)
                
                # Rename columns for clarity
                pivot_df = pivot_df.rename(columns={
                    'Unmatched_voter': 'Unmatched Donors',
                    'Partial_voter': 'Partial Donors',
                    'Full_voter': 'Full Donors',
                    'Unmatched_amountUSD': 'Unmatched Amount',
                    'Partial_amountUSD': 'Partial Matched Amount',
                    'Full_amountUSD': 'Full Matched Amount'
                })
                
                # Calculate totals and percentages
                pivot_df['Total Donors'] = pivot_df['Unmatched Donors'] + pivot_df['Partial Donors'] + pivot_df['Full Donors']
                pivot_df['Percent Donors Matched'] = (pivot_df['Partial Donors'] + pivot_df['Full Donors']) / pivot_df['Total Donors'] * 100
                
                # Round amount columns
                for col in ['Unmatched Amount', 'Partial Matched Amount', 'Full Matched Amount']:
                    pivot_df[col] = pivot_df[col].round(2)
                
                # Reorder columns
                column_order = ['project_name', 'Total Donors', 'Percent Donors Matched', 'Unmatched Donors', 'Partial Donors', 'Full Donors', 'Unmatched Amount', 'Partial Matched Amount', 'Full Matched Amount']
                pivot_df = pivot_df[column_order]
                
                # Sort by percent of donors matched in descending order
                pivot_df = pivot_df.sort_values('Percent Donors Matched', ascending=False)
                
                # Display the dataframe
                st.dataframe(
                    pivot_df,
                    column_config={
                        "project_name": "Project",
                        "Total Donors": st.column_config.NumberColumn("Total Donors", format="%d"),
                        "Percent Donors Matched": st.column_config.ProgressColumn("% Donors Matched", format="%.2f%%", min_value=0, max_value=100),
                        "Unmatched Donors": st.column_config.NumberColumn("Unmatched Donors", format="%d"),
                        "Partial Donors": st.column_config.NumberColumn("Partial Donors", format="%d"),
                        "Full Donors": st.column_config.NumberColumn("Full Donors", format="%d"),
                        "Unmatched Amount": st.column_config.NumberColumn("Unmatched Amount", format="$%.2f"),
                        "Partial Matched Amount": st.column_config.NumberColumn("Partial Matched Amount", format="$%.2f"),
                        "Full Matched Amount": st.column_config.NumberColumn("Full Matched Amount", format="$%.2f")
                    },
                    hide_index=True
                )

def calculate_matching_results(data):
    """Calculate matching results using different strategies (COCM and QF)."""
    # Apply voting eligibility based on passport scores and donation thresholds
    df_with_passport = fundingutils.apply_voting_eligibility(
        data['df'].copy(), 
        data['rounds'].get('min_donation_threshold_amount', 0).values[0],
        data['score_at_50_percent'],
        data['score_at_100_percent'],
        data['scaling_df']
    )
    donation_matrix = fundingutils.pivot_votes(df_with_passport)
    
    #donation_matrix.to_csv('name me'.csv')

    matching_cap_amount = data['matching_cap']
    matching_amount = data['matching_available']

    # Calculate matching amounts using both COCM and QF strategies
    matching_dfs = [fundingutils.get_qf_matching(strategy, donation_matrix, matching_cap_amount, matching_amount, cluster_df=donation_matrix, pct_cocm=data['pct_COCM']) 
                    for strategy in [data['strat'], 'QF']]


    # Merge results from both strategies
    matching_df = pd.merge(matching_dfs[0], matching_dfs[1], on='project_name', suffixes=(f'_{data["suffix"]}', '_QF'))
    
    # Add project details and calculate the difference between COCM and QF matching
    df_unique = data['df'][['project_name', 'chain_id', 'round_id', 'application_id']].drop_duplicates()
    matching_df = pd.merge(matching_df, df_unique, on='project_name', how='left')
    matching_df['Project Page'] = 'https://explorer.gitcoin.co/#/round/' + matching_df['chain_id'].astype(str) + '/' + matching_df['round_id'].astype(str) + '/' + matching_df['application_id'].astype(str)
    matching_df['Δ Match'] = matching_df[f'matching_amount_{data["suffix"]}'] - matching_df['matching_amount_QF']
    
    return matching_df.sort_values(f'matching_amount_{data["suffix"]}', ascending=False), donation_matrix

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
    
    # sorted_QF_amt = list(matching_df['matching_amount_QF']).sorted()
    # sorted_COCM_amt = list(matching_df[f'matching_amount_{s}']).sorted()

    qf_df = matching_df.copy()
    qf_df['Strategy'] = 'QF'
    qf_df['Amount'] = qf_df['matching_amount_QF']

    co_df = matching_df.copy()
    co_df['Strategy'] = s
    co_df['Amount'] = co_df[f'matching_amount_{s}']    

    plotly_df = pd.concat([qf_df, co_df])

    fig = px.histogram(plotly_df, x='project_name', y='Amount', color='Strategy', barmode='group')
    fig.update_layout(
        xaxis_title="Project Name",
        yaxis_title="Amount",
    )
    st.plotly_chart(fig)

    st.markdown(f'Matching Values shown above are in **{matching_token_symbol}**')


def display_singledonor_and_alldonor_stats(donation_matrix):
    with st.expander('Advanced: Stats on single donors and all-project donors'):
        st.write('''
            As a by-product of COCM\'s logic, the algorithm ignores donors who donate to only one project, 
            and donors who donate to every project. If projects are recieving less money than seems right under COCM, 
            the following stats on single and all-project donors may help.\n\n
            ''')
        single_and_all_df = pd.DataFrame(index = donation_matrix.columns, columns = ['# Donors passing Passport and low-donation checks', '# single-project donors', '# all-project donors'])
        single_and_all_df['# Donors passing Passport and low-donation checks'] = donation_matrix.apply(lambda p: sum(p.ne(0)))

        count_single_donors = lambda p : sum(donation_matrix.loc[i].ne(0)[p] and (sum(donation_matrix.loc[i].ne(0)) == 1) for i in donation_matrix.index)
        single_and_all_df['# single-project donors'] = donation_matrix.apply(lambda c: count_single_donors(c.name))

        single_and_all_df['# all-project donors'] = sum(donation_matrix.apply(all, axis=1))

        st.dataframe(
            single_and_all_df,
            use_container_width=True,
            hide_index=False
        )


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
    
    # Get projects in the round
    projects_df = get_project_summary_graphql(data['chain_id'], data['rounds']['round_id'].iloc[0])
    projects_df = projects_df[~projects_df['project_name'].isin(data['projects_to_remove'])]
    
    # CUSTOM RULE: Remove application ID 90 from round 608, duplicate project
    if data['rounds']['round_id'].iloc[0] == '608' and data['chain_id'] == 42161:
        if '90' in projects_df['id'].values:
            projects_df = projects_df[projects_df['id'] != '90']
    
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
    """Adjust matching funds if there's an overflow, using Decimal for high precision and handling numpy types."""
    getcontext().prec = 36  # Set precision high enough to handle 18 decimal places safely

    # Convert numpy types to Python types
    matching_funds_available = float(matching_funds_available)
    matching_token_decimals = int(matching_token_decimals)

    full_matching_funds_available = Decimal(str(matching_funds_available)) * Decimal(10**matching_token_decimals)
    total_matched = sum(Decimal(str(x)) for x in output_df['matched'])
    matching_overflow = total_matched - full_matching_funds_available
    
    if matching_overflow <= 0:
        return output_df
    #st.warning(f'Initial Matching Overflow: {matching_overflow / Decimal(10**matching_token_decimals)}. Adjusting Matching Funds')
    
    # Calculate the reduction factor
    reduction_factor = full_matching_funds_available / total_matched
    # Apply the reduction factor to all matched amounts
    output_df['matched'] = output_df['matched'].apply(lambda x: int(Decimal(str(x)) * reduction_factor))

    # Distribute any remaining overflow due to rounding
    remaining_overflow = sum(Decimal(str(x)) for x in output_df['matched']) - full_matching_funds_available
    if remaining_overflow > 0:
        sorted_indices = output_df['matched'].argsort()[::-1]
        for i in range(int(remaining_overflow)):
            output_df.iloc[sorted_indices[i], output_df.columns.get_loc('matched')] -= 1

    final_overflow = sum(Decimal(str(x)) for x in output_df['matched']) - full_matching_funds_available
    #st.success(f'Matching funds adjusted in 1 iteration. Final overflow: {final_overflow / Decimal(10**matching_token_decimals)}')
    if final_overflow > 0:
        st.error('There is a matching overflow. Please contact the team for further assistance.')

    return output_df

def display_matching_distribution(output_df):
    """Display and provide download option for the matching distribution."""
    # Create a copy for display purposes
    display_df = output_df.copy()
    
    # Format large numbers for display
    for col in ['matched', 'totalReceived', 'sumOfSqrt', 'capOverflow', 'matchedWithoutCap']:
        display_df[col] = display_df[col].apply(lambda x: f"{x}")
    
    st.dataframe(display_df, use_container_width=True)
    
    # Use the original output_df for CSV download
    st.download_button(
        label="⬇ Download Matching Distribution",
        data=output_df.to_csv(index=False),
        file_name='matching_distribution.csv',
        mime='text/csv'
    )
    st.write('You can upload this CSV to manager.gitcoin.co to apply the matching results to your round.')
    st.write('Note: If manual edits are needed, the key column to update is "matched". Values must be integers without decimals or commas, as our contracts expect token amounts in their smallest unit (e.g., wei for ETH). Incorrect formatting could cause errors in fund allocation.')

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
    
    check_round_existence(round_id, chain_id)

    # Advanced options 

    filterin_df=None
    filterout_df=None
    arbitrary_df=None
    scaling_df=None
    with st.expander("Advanced: Override Passport Scaling"):

        
        if st.toggle('Filter in wallets', value=False, key='filterin-toggle'):
            filterout_df = handle_csv_upload(purpose='filter in')

        if st.toggle('Filter out wallets',value=False,key='filterout-toggle'):
            filterin_df = handle_csv_upload(purpose='filter out')

        if st.toggle('Arbitrary scaling (e.g. Tunable QF)',value=False,key='arbitraryscale-toggle'):
            arbitrary_df = handle_csv_upload(purpose='general scaling')

    uploaded_dfs = [x for x in [filterin_df, filterout_df, arbitrary_df] if x is not None]
    if len(uploaded_dfs) >= 1:
        scaling_df = pd.concat(uploaded_dfs)

    # half_and_half = False
    # with st.expander("Advanced: Give results as half COCM / half QF"):
    #     st.write('''
    #         Toggle the switch below to calculate a matching result blending COCM and QF, instead of pure COCM.
    #         In this case, funding amounts will be found for each mechanism, normalized to the size of the matching pool, and then averaged for each project.
    #         E.g. if a project gets 10% of the matching pool under COCM and 40% of the matching pool under QF, they will get 25% of the matching pool under this calculation.''')
    #     half_and_half = st.toggle('Select for half-and-half')

    pct = 1
    with st.expander('Advanced: Blend COCM and QF'):
        st.write('''Use the slider below to blend COCM and QF together in your results. 
                    Funding amounts will be found for each mechanism, normalized to the size of the matching pool, and then blended for each project. 
                    Set the slider to "1" to just use COCM.
                    Pure QF results are always available separately.''')
        c1,c2,c3 = st.columns(3)
        pct = c1.slider('Percent COCM', min_value = 0.25, max_value=1.0, value=1.0, step=0.25)
    
    # Load and process data
    data = load_data(round_id, chain_id)
    data['scaling_df'] = scaling_df

    if pct == 1:
        data['strat'] = 'COCM'
        data['suffix'] = 'COCM'
        data['pct_COCM'] = 1
    else:
        data['strat'] = 'pctCOCM'
        data['suffix'] = (str(pct)+'0')[2:4]+'pctCOCM'
        data['pct_COCM'] = pct


    # matching_pool_size_override = None
    # with st.expander('Advanced: Change Matching Pool Size'):
    #     matching_pool_size_override = st.number_input(f"New matching pool size (in {data['config_df']['token_code'].iloc[0]}): ", value=float(data['matching_available']), min_value=1.0)
    # if matching_pool_size_override is not None:
    #     data['matching_available'] = matching_pool_size_override
    # # Display various settings of the round

    
    display_round_settings(data)


    matching_amount = data['matching_available']
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
    matching_df, donation_matrix = calculate_matching_results(data)
    display_matching_results(matching_df, data['config_df']['token_code'].iloc[0], data['suffix'])
    display_singledonor_and_alldonor_stats(donation_matrix)
    display_network_graph(data['df'])


    # Matching distribution download section
    st.subheader('Download Matching Distribution')
    if calculate_percent_scored_voters(data) < 99:
        st.warning('Matching distribution download is not recommended until more addresses are scored. This could take 24-72 hours after the round concludes.')
    strategy_choice = select_matching_strategy(data['suffix'])
    output_df = prepare_output_dataframe(matching_df, strategy_choice, data)
    output_df = adjust_matching_overflow(output_df, data['matching_available'], data['config_df']['token_decimals'].iloc[0])
    display_matching_distribution(output_df)
    with st.expander('Advanced: Matching Overflow Information', expanded=False):
        try:
            full_matching_funds_available = Decimal(str(data['matching_available'])) * Decimal(10**int(data['config_df']['token_decimals'].iloc[0]))
            total_matched = sum(Decimal(str(x)) for x in output_df['matched'])
            matching_overflow = total_matched - full_matching_funds_available
            st.write(f" Matching Overflow: {(matching_overflow / Decimal(10**int(data['config_df']['token_decimals'].iloc[0]))):.18f} or {matching_overflow} wei")
        except (ValueError, TypeError, decimal.InvalidOperation) as e:
            st.error(f"Error calculating matching overflow: {str(e)}")
    # Summary section
    st.header('📈 Sharable Summary')
    token_code = data['config_df']['token_code'].iloc[0]
    summary_df = create_summary_dataframe(output_df, matching_df, token_code, data['suffix'])
    display_summary(summary_df)

    #matching_distribution_chart = create_matching_distribution_chart(summary_df, token_code)
    #st.plotly_chart(matching_distribution_chart)

if __name__ == "__main__":
    main()