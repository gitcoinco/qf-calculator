SELECT
  ("public"."Round"."metadata" #>> '{name}')::text AS "round_name",
  "public"."Round"."amountUSD" AS "amountUSD",
  "public"."Round"."votes" AS "votes",
  "public"."Round"."uniqueContributors" AS "uniqueContributors",
  --TIMESTAMP 'epoch' + ("public"."Round"."roundStartTime") * INTERVAL '1 second' as "round_start_time",
  --TIMESTAMP 'epoch' + ("public"."Round"."roundEndTime") * INTERVAL '1 second' as "round_end_time",
  "public"."Round"."chainId" AS "chain_id",
  "public"."Round"."roundId" AS "round_id",
  ("public"."Round"."metadata" #>> ARRAY['quadraticFundingConfig', 'matchingCap'])::boolean AS "has_matching_cap",
  ("public"."Round"."metadata" #>> ARRAY['quadraticFundingConfig', 'matchingCapAmount'])::double precision AS "matching_cap_amount",
  ("public"."Round"."metadata" #>> ARRAY['quadraticFundingConfig', 'matchingFundsAvailable'])::double precision AS "matching_funds_available",
  "public"."Round"."token" AS "token",
  ("public"."Round"."metadata" #>> ARRAY['quadraticFundingConfig', 'minDonationThreshold'])::boolean AS "has_min_donation_threshold",
  ("public"."Round"."metadata" #>> ARRAY['quadraticFundingConfig', 'minDonationThresholdAmount'])::double precision AS "min_donation_threshold_amount",
  ("public"."Round"."metadata" #>> ARRAY['quadraticFundingConfig', 'sybilDefense'])::text AS "sybilDefense"
FROM
  "public"."Round" 
WHERE 
        ("public"."Round"."metadata" ->> 'name')::text NOT LIKE '%test%' and 
        ("public"."Round"."metadata" ->> 'name')::text NOT LIKE '%Test%' and
        ("public"."Round"."metadata" #>> ARRAY['support', 'info'])::varchar NOT IN ('meg@gmail.com', 'test@gitcoin.co') and
        "public"."Round"."amountUSD" > 0
ORDER BY
  "public"."Round"."amountUSD" DESC

