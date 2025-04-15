from datetime import datetime, timezone
import streamlit as st
import pandas as pd
import requests

# Cache TTL values
ttl_short = 900  # 15 minutes

@st.cache_resource(ttl=ttl_short)
def get_recent_rounds_graphql(limit=100, offset=0):
    """
    Fetch recent rounds data using the Gitcoin indexer GraphQL API.
    
    Parameters:
    -----------
    limit : int, optional
        Maximum number of records to return (default: 100)
    offset : int, optional
        Number of records to skip (default: 0)
    custom_filters : dict, optional
        Dictionary of custom filter conditions to apply
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing the filtered and paginated round data
    """
    # Get current time in ISO 8601 format with timezone
    current_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S%z")
    # Format the timezone part to match the expected format (e.g., +00:00 instead of +0000)
    current_time = current_time[:-2] + ":" + current_time[-2:]
    
    # GraphQL query
    query = """
    query GetRecentRounds($limit: Int, $offset: Int, $currentTime: timestamptz) {
        rounds(
            where: {
                strategyName: {_eq: "allov2.DonationVotingMerkleDistributionDirectTransferStrategy"},
                donationsEndTime: {_lte: $currentTime}
                uniqueDonorsCount: {_gte: 10}
            },
            limit: $limit, 
            offset: $offset,
            orderBy: {
                donationsEndTime: DESC_NULLS_LAST
            }
        ) {
            id
            chainId
            matchTokenAddress
            matchAmountInUsd
            matchAmount
            totalDonationsCount
            uniqueDonorsCount
            totalAmountDonatedInUsd
            roundMetadata
            donationsStartTime
            donationsEndTime
        }
    }
    """
    
    # Prepare variables for the query
    variables = {
        "limit": limit,
        "offset": offset,
        "currentTime": current_time
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
        
        # Extract rounds data
        if "data" in data and "rounds" in data["data"]:
            rounds_data = data["data"]["rounds"]
            
            # Convert to DataFrame
            if rounds_data:
                # Extract nested fields from roundMetadata
                processed_data = []
                for round in rounds_data:
                    round_dict = {
                        "round_id": round.get("id"),
                        "chain_id": round.get("chainId"),
                        "amountUSD": round.get("totalAmountDonatedInUsd"),
                        "votes": round.get("totalDonationsCount"),
                        "uniqueContributors": round.get("uniqueDonorsCount"),
                        "donations_start_time": round.get("donationsStartTime"),
                        "donations_end_time": round.get("donationsEndTime"),
                        "token": round.get("matchTokenAddress"),
                    }
                    
                    # Extract fields from roundMetadata if available
                    if round.get("roundMetadata"):
                        metadata = round["roundMetadata"]
                        round_dict["round_name"] = metadata.get("name")
                        round_dict["has_matching_cap"] = metadata.get("quadraticFundingConfig", {}).get("matchingCap")
                        round_dict["matching_cap_amount"] = metadata.get("quadraticFundingConfig", {}).get("matchingCapAmount")
                        round_dict["matching_funds_available"] = metadata.get("quadraticFundingConfig", {}).get("matchingFundsAvailable")
                        round_dict["has_min_donation_threshold"] = metadata.get("quadraticFundingConfig", {}).get("minDonationThreshold")
                        round_dict["min_donation_threshold_amount"] = metadata.get("quadraticFundingConfig", {}).get("minDonationThresholdAmount", 0)
                        round_dict["sybilDefense"] = metadata.get("quadraticFundingConfig", {}).get("sybilDefense")

                    processed_data.append(round_dict)
                
                df = pd.DataFrame(processed_data)
                
                return df
            else:
                return pd.DataFrame()
        else:
            print("No data or rounds found in response")
            st.warning("No data returned from the GraphQL API")
            return pd.DataFrame()
            
    except requests.RequestException as e:
        print(f"Request exception: {e}")
        st.error(f"Error fetching data from GraphQL API: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Unexpected error: {e}")
        st.error(f"Unexpected error: {e}")
        return pd.DataFrame() 