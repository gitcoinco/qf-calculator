import streamlit as st
import pandas as pd
import requests
import json

# Cache TTL values
ttl_short = 900  # 15 minutes

@st.cache_resource(ttl=ttl_short)
def get_project_summary_graphql(chain_id, round_id, limit=200, offset=0):
    """
    Fetch project summary data for a specific round using the Gitcoin indexer GraphQL API.
    This function will automatically paginate through all results.

    Parameters:
    -----------
    chain_id : int
        The chain ID of the round
    round_id : str
        The round ID to fetch applications for
    limit : int, optional
        Maximum number of records to return per request (default: 100)
    offset : int, optional
        Number of records to skip (default: 0)

    Returns:
    --------
    pandas.DataFrame
        DataFrame containing all applications for the specified round
    """
    # GraphQL query
    query = """
    query GetProjectSummary($chainId: Int!, $roundId: String!, $limit: Int!, $offset: Int!) {
        applications(
            where: {
                chainId: {_eq: $chainId},
                roundId: {_eq: $roundId},
                status: {_eq: "APPROVED"}
            },
            limit: $limit,
            offset: $offset
        ) {
            id
            chainId
            roundId
            projectId
            status
            totalAmountDonatedInUsd
            uniqueDonorsCount
            totalDonationsCount
            metadata
        }
    }
    """

    # Initialize variables for pagination
    all_applications = []
    has_more = True
    current_offset = offset

    # Loop until we've fetched all applications
    while has_more:
        # Prepare variables for the query
        variables = {
            "chainId": chain_id,
            "roundId": round_id,
            "limit": limit,
            "offset": current_offset
        }

        # Make the GraphQL request
        try:
            response = requests.post(
                "https://beta.indexer.gitcoin.co/v1/graphql",
                json={"query": query, "variables": variables},
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            data = response.json()

            # Extract applications data
            if "data" in data and "applications" in data["data"]:
                applications_data = data["data"]["applications"]

                # Add to our collection
                all_applications.extend(applications_data)

                # Check if we need to continue pagination
                if len(applications_data) < limit:
                    has_more = False
                else:
                    current_offset += limit
            else:
                has_more = False

        except requests.RequestException as e:
            print(f"Request exception: {e}")
            st.error(f"Error fetching data from GraphQL API: {e}")
            has_more = False
        except Exception as e:
            print(f"Unexpected error: {e}")
            st.error(f"Unexpected error: {e}")
            has_more = False

    # Process the data into a DataFrame
    if all_applications:
        processed_data = []
        for app in all_applications:
            app_dict = {
                "id": app.get("id"),
                "chain_id": app.get("chainId"),
                "round_id": app.get("roundId"),
                "project_id": app.get("projectId"),
                "status": app.get("status"),
                "total_amount_donated_in_usd": app.get("totalAmountDonatedInUsd"),
                "unique_donors_count": app.get("uniqueDonorsCount"),
                "total_donations_count": app.get("totalDonationsCount")
            }

            # Extract project name and recipient address from metadata
            if app.get("metadata"):
                metadata = app["metadata"]
                app_dict["project_name"] = metadata.get("application", {}).get("project", {}).get("title")
                app_dict["recipient_address"] = metadata.get("application", {}).get("recipient")

            processed_data.append(app_dict)

        df = pd.DataFrame(processed_data)
        return df
    else:
        return pd.DataFrame()
