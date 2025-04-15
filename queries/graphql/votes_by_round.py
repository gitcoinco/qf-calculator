import streamlit as st
import pandas as pd
import requests
import json

# Cache TTL values
ttl_short = 900  # 15 minutes

@st.cache_resource(ttl=ttl_short)
def get_votes_by_round_graphql(chain_id, round_id, limit=200, offset=0):
    """
    Fetch all donations for a specific round using the Gitcoin indexer GraphQL API.
    This function will automatically paginate through all results.
    
    Parameters:
    -----------
    chain_id : int
        The chain ID of the round
    round_id : str
        The round ID to fetch donations for
    limit : int, optional
        Maximum number of records to return per request (default: 200)
    offset : int, optional
        Number of records to skip (default: 0)
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing all donations for the specified round
    """
    # GraphQL query
    query = """
    query GetVotesByRound($chainId: Int!, $roundId: String!, $limit: Int!, $offset: Int!) {
        donations(
            where: {
                chainId: {_eq: $chainId}, 
                roundId: {_eq: $roundId},
                application: {status: {_eq: "APPROVED"}}
            },
            orderBy: {timestamp: DESC}
            limit: $limit, 
            offset: $offset
        ) {
            id
            chainId
            roundId
            projectId
            recipientAddress
            donorAddress
            amount
            amountInUsd
            amountInRoundMatchToken
            tokenAddress
            blockNumber
            transactionHash
            application {
                id
                metadata
            }
        }
    }
    """
    
    # Initialize variables for pagination
    all_donations = []
    has_more = True
    current_offset = offset
    
    # Loop until we've fetched all donations
    while has_more:
        # Prepare variables for the query
        variables = {
            "chainId": chain_id,
            "roundId": round_id,
            "limit": limit,
            "offset": current_offset
        }

        print(variables)
        
        # Make the GraphQL request
        try:
            response = requests.post(
                "https://beta.indexer.gitcoin.co/v1/graphql",
                json={"query": query, "variables": variables},
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            data = response.json()
            
            # Extract donations data
            if "data" in data and "donations" in data["data"]:
                donations_data = data["data"]["donations"]
                
                # Add to our collection
                all_donations.extend(donations_data)
                
                # Check if we need to continue pagination
                if len(donations_data) < limit:
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
    if all_donations:
        # with open("all_donations.txt", "w") as f:
        #     f.write(json.dumps(all_donations, indent=2))
        processed_data = []
        for donation in all_donations:
            donation_dict = {
                "id": donation.get("id"),
                "chain_id": donation.get("chainId"),
                "round_id": donation.get("roundId"),
                "project_id": donation.get("projectId"),
                "recipient_address": donation.get("recipientAddress").lower(),
                "voter": donation.get("donorAddress").lower(),
                "amount": donation.get("amount"),
                "amountUSD": donation.get("amountInUsd"),
                "amount_in_round_match_token": donation.get("amountInRoundMatchToken"),
                "token_address": donation.get("tokenAddress").lower(),
                "block_number": donation.get("blockNumber"),
                "transaction_hash": donation.get("transactionHash"),
                "application_id": donation.get("application", {}).get("id")
            }

            donation_dict["project_name"] = (
                donation.get("application", {}).get("metadata", {})
                        .get("application", {})
                        .get("project", {})
                        .get("title")
            )
            
            processed_data.append(donation_dict)
        
        # with open("processed_data.txt", "w") as f:
        #     f.write(json.dumps(processed_data, indent=2))

        df = pd.DataFrame(processed_data)
        return df
    else:
        return pd.DataFrame()
