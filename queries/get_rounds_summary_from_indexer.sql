SELECT
  ("chain_data_3287eeeb342085_62"."rounds"."round_metadata" #>> array['name']::text[])::text AS "round_name",
  "chain_data_3287eeeb342085_62"."rounds"."total_amount_donated_in_usd" AS "amountUSD",
  "chain_data_3287eeeb342085_62"."rounds"."total_donations_count" AS "votes",
  "chain_data_3287eeeb342085_62"."rounds"."unique_donors_count" AS "uniqueContributors",
  "chain_data_3287eeeb342085_62"."rounds"."chain_id" AS "chain_id",
  "chain_data_3287eeeb342085_62"."rounds"."id" AS "round_id",
  ("chain_data_3287eeeb342085_62"."rounds"."round_metadata" #>> ARRAY['quadraticFundingConfig', 'matchingCap'])::boolean AS "has_matching_cap",
  ("chain_data_3287eeeb342085_62"."rounds"."round_metadata" #>> ARRAY['quadraticFundingConfig', 'matchingCapAmount'])::double precision AS "matching_cap_amount",
  ("chain_data_3287eeeb342085_62"."rounds"."round_metadata" #>> ARRAY['quadraticFundingConfig', 'matchingFundsAvailable'])::double precision AS "matching_funds_available",
  "chain_data_3287eeeb342085_62"."rounds"."match_token_address" AS "token",
  ("chain_data_3287eeeb342085_62"."rounds"."round_metadata" #>> ARRAY['quadraticFundingConfig', 'minDonationThreshold'])::boolean AS "has_min_donation_threshold",
  ("chain_data_3287eeeb342085_62"."rounds"."round_metadata" #>> ARRAY['quadraticFundingConfig', 'minDonationThresholdAmount'])::double precision AS "min_donation_threshold_amount",
  ("chain_data_3287eeeb342085_62"."rounds"."round_metadata" #>> ARRAY['quadraticFundingConfig', 'sybilDefense'])::text AS "sybilDefense"
FROM
  "chain_data_3287eeeb342085_62"."rounds"
WHERE
  ("chain_data_3287eeeb342085_62"."rounds"."round_metadata" #>> array['name']::text[])::text NOT LIKE '%test%'
  AND ("chain_data_3287eeeb342085_62"."rounds"."round_metadata" #>> array['name']::text[])::text NOT LIKE '%Test%'
  AND ("chain_data_3287eeeb342085_62"."rounds"."round_metadata" #>> ARRAY['support', 'info'])::varchar NOT IN ('meg@gmail.com', 'test@gitcoin.co')
