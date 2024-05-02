SELECT
    (r."round_metadata" #>> '{name}')::text AS "round_name",
    r."total_amount_donated_in_usd" AS "amountUSD",
    r."total_donations_count" AS "votes",
    r."unique_donors_count" AS "uniqueContributors",
    r."chain_id",
    r."id" AS "round_id",
    (r."round_metadata" #>> '{quadraticFundingConfig, matchingCap}')::boolean AS "has_matching_cap",
    (r."round_metadata" #>> '{quadraticFundingConfig, matchingCapAmount}')::double precision AS "matching_cap_amount",
    (r."round_metadata" #>> '{quadraticFundingConfig, matchingFundsAvailable}')::double precision AS "matching_funds_available",
    r."match_token_address" AS "token",
    (r."round_metadata" #>> '{quadraticFundingConfig, minDonationThreshold}')::boolean AS "has_min_donation_threshold",
    (r."round_metadata" #>> '{quadraticFundingConfig, minDonationThresholdAmount}')::double precision AS "min_donation_threshold_amount",
    (r."round_metadata" #>> '{quadraticFundingConfig, sybilDefense}')::text AS "sybilDefense"
FROM
    "chain_data_63"."rounds" AS r
