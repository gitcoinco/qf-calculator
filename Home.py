import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
import utils
import fundingutils

# Page configuration
st.set_page_config(page_title="Matching Results Calculator", page_icon="assets/favicon.png", layout="centered")




def display_round_statistics(df):
    """Display some statistics for the current round."""
    st.subheader(f"Round Stats")
    col1, col2 = st.columns(2)
    col1.write(f"**Total Amount Raised:** {df['amount'].sum():,.2f}")
    col1.write(f"**Number of Unique Voters:** {df['voter'].nunique()}")
    col2.write(f"**Number of Unique Projects:** {df['project_name'].nunique()}")
    col2.write(f"**Avg. Projects per Voter:** {df.groupby('voter')['project_name'].nunique().mean():.2f}")

def display_crowdfunding_stats(df, matching_funds_available):
    """Display crowdfunding statistics and metrics."""
    st.header('👥 Crowdfunding')
    crowd_raised = df['amount'].sum()
    
    grouped_voter_data = df.groupby('voter')['amount'].agg(['sum', 'mean', 'median', 'max']).reset_index()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Median Donor Contribution", f"{grouped_voter_data['median'].median():.2f}")
    col2.metric("Average Donor Contribution", f"{grouped_voter_data['mean'].mean():.2f}")
    col3.metric("Max Donor Contribution", f"{grouped_voter_data['max'].max():.2f}")

    return grouped_voter_data

def create_donation_distribution_chart(grouped_voter_data):
    """Create a chart showing the distribution of donor contributions."""
    bin_edges = [0, 1, 2, 3, 4, 5, 10, 20, 30, 50, 100, 500, 1000, np.inf]
    bin_labels = ['0-1','1-2','2-3','3-4', '4-5', '5-10', '10-20','20-30','30-50', '50-100', '100-500', '500-1000', '1000+']
    grouped_voter_data['amountUSD_bin'] = pd.cut(grouped_voter_data['sum'], bins=bin_edges, labels=bin_labels, right=False)
    
    fig = px.histogram(grouped_voter_data, x="amountUSD_bin", category_orders={'amountUSD_bin': bin_labels},
                       labels={'amountUSD_bin': 'Donation Amount Range'}, nbins=len(bin_edges)-1)
    fig.update_traces(hovertemplate='<b>Donation Range:</b> $%{x}<br><b>Number of Donors:</b> %{y}')
    fig.update_layout(
        title_text='Distribution of Donor Contributions by Amount',
        xaxis=dict(title='Donation Amount Range', titlefont=dict(size=18), tickfont=dict(size=14)),
        yaxis=dict(title='Number of Donors', titlefont=dict(size=18), tickfont=dict(size=14)),
        bargap=0.3
    )
    return fig

def calculate_matching_results(df, matching_cap_amount, matching_funds_available ):
    """Calculate matching results using different strategies (COCM and QF)."""
    # Apply voting eligibility based on passport scores and donation thresholds
    
    votes_df = fundingutils.pivot_votes(df)

    
    # Calculate matching amounts using both COCM and QF strategies
    matching_dfs = [fundingutils.get_qf_matching(strategy, votes_df, matching_cap_amount, matching_funds_available, cluster_df=votes_df) 
                    for strategy in ['COCM', 'QF']]
    
    # Merge results from both strategies
    matching_df = pd.merge(matching_dfs[0], matching_dfs[1], on='project_name', suffixes=('_COCM', '_QF'))
    matching_df['Δ Match'] = matching_df['matching_amount_COCM'] - matching_df['matching_amount_QF']
    
    return matching_df.sort_values('matching_amount_COCM', ascending=False)

def display_matching_results(matching_df):
    """Display the matching results in a formatted table."""
    st.subheader('Results Comparison')
    column_config = {
        "project_name": st.column_config.TextColumn("Project"),
        "matching_amount_COCM": st.column_config.NumberColumn("COCM Match", format="%.2f"),
        "matching_amount_QF": st.column_config.NumberColumn("QF Match", format="%.2f"),
        "Δ Match": st.column_config.NumberColumn("Δ Match", format="%.2f")
    }
    
    display_columns = ['project_name', 'matching_amount_COCM', 'matching_amount_QF', 'Δ Match']
    st.dataframe(
        matching_df[display_columns],
        use_container_width=True,
        column_config=column_config,
        hide_index=True
    )
    
    #st.markdown(f'Matching Values shown above are in **{matching_token_symbol}**')

def select_matching_strategy():
    """Allow user to select the matching strategy for download."""
    return st.selectbox(
        'Select the matching strategy to download:',
        ('COCM', 'QF')
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
    
    if matching_overflow >= 0:
        st.warning('Potential Matching Overflow Detected. Adjusting Matching Funds')
        matching_adjustment = int(matching_overflow / max(output_df['matched'].count(), 1))
        output_df['matched'] = output_df['matched'].apply(lambda x: str(max(int(x) - matching_adjustment, 0)))
        matching_overflow = sum(int(x) for x in output_df['matched']) - full_matching_funds_available
        st.warning(f'Adjusted Matching Overflow is {matching_overflow}') # IF THIS NUMBER IS NEGATIVE WE ARE GOOD TO GO
    
    return output_df

def display_matching_distribution(output_df):
    """Display and provide download option for the matching distribution."""
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
    st.header(f'The value of the sum of the matched column is {output_df["matched"].sum()}')

def create_summary_dataframe(output_df, matching_df, token_code):
    """Create a summary dataframe for the round results."""
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

def create_voter_overlap_heatmap(df):
    df['Project_Short'] = df['project_name'].apply(lambda x: x[:15] + '...' if len(x) > 15 else x)

    titles = df['Project_Short'].unique().tolist()
    overlap_matrix = pd.DataFrame(0, index=titles, columns=titles)

    for i in range(len(titles)):
        for j in range(i+1, len(titles)):
            title1, title2 = titles[i], titles[j]

            voters1 = set(df[df['Project_Short'] == titles[i]]['voter'])
            voters2 = set(df[df['Project_Short'] == titles[j]]['voter'])

            common_voters = len(voters1.intersection(voters2))
            total_voters = len(voters1.union(voters2))

            overlap_percentage = common_voters / total_voters * 100
            overlap_matrix.at[title1, title2] = overlap_percentage
            overlap_matrix.at[title2, title1] = overlap_percentage

    # Flatten the overlap matrix and sort by overlap percentage in descending order
    overlap_matrix_flat = overlap_matrix.stack().sort_values(ascending=False)

    # Select the top 20 projects
    unique_projects = []
    for idx, overlap in overlap_matrix_flat.items():
        if idx[0] not in unique_projects:
            unique_projects.append(idx[0])
        if idx[1] not in unique_projects:
            unique_projects.append(idx[1])
        if len(unique_projects) >= 20:
            break

    # Create a DataFrame from the top overlaps using the complete overlap matrix
    top_overlaps_df = overlap_matrix.loc[unique_projects, unique_projects]

    # Create a heatmap using plotly
    fig = go.Figure(data=go.Heatmap(
                    z=top_overlaps_df.values,
                    x=top_overlaps_df.columns,
                    y=top_overlaps_df.index,
                    hoverongaps = False,
                    colorscale='reds',
                    hovertemplate = '<i>Overlap</i>: %{z:.2f}%<extra></extra>'
    ))

    fig.update_layout( xaxis_nticks=36)
    return fig, overlap_matrix

def main():
    """Main function to run the Streamlit app."""
    st.image('assets/657c7ed16b14af693c08b92d_GTC-Logotype-Dark.png', width=300)
    
    st.title('Matching Results Calculator')
    st.write('You can use this tool to calculate and compare matching results under a variety of quadratic funding variants. This tool is for when the data is in a CSV and not easily queryable.')
    st.write('Simply enter your round settings then upload a csv file with three columns to get started: voter, project_name, and amount')
    st.write('Voter should be the unique id of the voter (usually a wallet address), project_name should be the unique name of the project, and amount should be the amount donated (usually in USD).')

    # Accept inputs for matching_funds_available, matching_token_price, and matching_cap
    st.subheader('Enter Round Settings')
    col1, col2 = st.columns(2)
    matching_funds_available = col1.number_input(
        'Matching Funds Available in Token',
        min_value=0.0,
        value=10000.0,
        step=100.0,
        format="%.2f"
    )
    matching_cap = col2.number_input(
        'Matching Cap (%)',
        min_value=0.0,
        max_value=100.0,
        value=10.0,
        step=1.0,
        format="%.2f"
    )
    #matching_token_decimals = col2.number_input(
    #    'Matching Token Decimals',
    #    min_value=0,
    #    value=18,
    #    step=1
    #)

    csv = st.file_uploader('Upload a CSV file with the required columns', type=['csv'])
    if csv is None:
        st.stop()
    df = pd.read_csv(csv)
    df.head()
    display_round_statistics(df)
    
    # Crowdfunding stats section
    grouped_voter_data = display_crowdfunding_stats(df, matching_funds_available)
    st.plotly_chart(create_donation_distribution_chart(grouped_voter_data), use_container_width=True)


    # Quadratic Funding Method Comparison section
    st.header('💚 Quadratic Funding Method Comparison')
    st.write('''Quadratic funding helps us solve coordination failures by creating a way for community members to fund what matters to them while amplifying their impact. However, its assumption that people make independent decisions can be exploited to unfairly influence the distribution of matching funds.''')
    st.write('''Connection-oriented cluster-matching (COCM) doesn't make this assumption. Instead, it quantifies just how coordinated groups of actors are likely to be based on the social signals they have in common. Projects backed by more independent agents receive greater matching funds. Conversely, if a project's support network shows higher levels of coordination, the matching funds are reduced, encouraging self-organized solutions within more coordinated groups. ''')
    # Allow removal of projects from matching distribution
    all_projects = df['project_name'].unique()
    projects_to_remove = st.multiselect('Projects may be removed from the matching distribution by selecting them here:', all_projects)
    df = df[~df['project_name'].isin(projects_to_remove)]
    # Calculate and display matching results
    matching_df = calculate_matching_results(df, matching_cap, matching_funds_available)
    display_matching_results(matching_df)

    total_delta_sum = matching_df['Δ Match'].abs().sum()
    st.metric("Sum of Absolute Value of Δ Match", f"{total_delta_sum:.2f}")
    
    percent_redistributed = total_delta_sum / matching_funds_available
    st.metric("Percent of Matching Funds Redistributed", f"{percent_redistributed:.2%}")

    zuzalu_q1_results = pd.read_csv('data/zuzalu_q1_redistributions.csv')
    filtered_results = zuzalu_q1_results[(zuzalu_q1_results['voter_set'] == 'All') & (zuzalu_q1_results['algo'] == 'COCM')]
    tech_redistribution_pct = filtered_results[filtered_results['round'] == 'Tech']['pct'].iloc[0]
    events_redistribution_pct = filtered_results[filtered_results['round'] == 'Events']['pct'].iloc[0]

    poap_results = zuzalu_q1_results[(zuzalu_q1_results['voter_set'] == 'Zupass') & (zuzalu_q1_results['algo'] == 'COCM')]
    poap_tech_redistribution_pct = poap_results[poap_results['round'] == 'Tech']['pct'].iloc[0]
    poap_events_redistribution_pct = poap_results[poap_results['round'] == 'Events']['pct'].iloc[0]
    
    st.write('')
    st.write('To better understand how strong of an effect COCM would have on the results of this round, the below graph vizualizes how this redistribution amount compares to Zuzalu Q1 Rounds.')

    fig = go.Figure(data=[
        go.Bar(name='All Voters', x=['Tech', 'Events'], y=[tech_redistribution_pct * 100, events_redistribution_pct * 100]),
        go.Bar(name='Zupass Voters', x=['Tech', 'Events'], y=[poap_tech_redistribution_pct * 100, poap_events_redistribution_pct * 100])
    ])

    fig.add_shape(
        type='line',
        x0=-0.5,
        y0=percent_redistributed * 100,
        x1=1.5,
        y1=percent_redistributed * 100,
        line=dict(
            color='Red',
            width=2,
            dash='dashdot'
        ),
        name='This Round'
    )

    fig.update_layout(
        title='Zuzalu Q1 Redistribution Percentages',
        xaxis_title='Voter Set',
        yaxis_title='Redistribution Percent',
        barmode='group'
    )

    fig.add_annotation(
        x=0.5,
        y=percent_redistributed * 100,
        text=f'This Round: {percent_redistributed * 100:.2f}%',
        showarrow=True,
        arrowhead=1,
        font=dict(color='red')
    )

    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader('Comparing the Zuzalu Q1 redistribution to the results of this round:')
    tech_redistribution_diff = percent_redistributed - tech_redistribution_pct
    events_redistribution_diff = percent_redistributed - events_redistribution_pct

   
    # Visualize the difference between QF and COCM
    st.subheader('📊 Top Voter Overlaps per Pair of Project')
    st.write('If there is collusion-like behavior, can we identify where its strongest?')
    voter_overlap_heatmap, overlap_matrix = create_voter_overlap_heatmap(df)
    st.plotly_chart(voter_overlap_heatmap)

    # Matching distribution download section
    #st.subheader('Download Matching Distribution')
    #strategy_choice = select_matching_strategy()
    #output_df = prepare_output_dataframe(matching_df, strategy_choice, data)
    #output_df = adjust_matching_overflow(output_df, data['rounds']['matching_funds_available'].values[0], data['config_df']['token_decimals'].iloc[0])
    #display_matching_distribution(output_df)

    # Summary section
    #st.header('📈 Sharable Summary')
    #token_code = data['config_df']['token_code'].iloc[0]
    #summary_df = create_summary_dataframe(output_df, matching_df, token_code)
    #display_summary(summary_df)


    #matching_distribution_chart = create_matching_distribution_chart(summary_df, token_code)
    #st.plotly_chart(matching_distribution_chart)

if __name__ == "__main__":
    main()