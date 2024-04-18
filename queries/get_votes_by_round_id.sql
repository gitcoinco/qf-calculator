WITH FilteredRounds AS (
    SELECT
    (r."metadata" ->> 'name')::text AS "round_name",
      r."amountUSD" AS "amountUSD",
      r."votes" AS "votes",
      r."uniqueContributors" AS "uniqueContributors",
      TIMESTAMP 'epoch' + (r."roundStartTime") * INTERVAL '1 second' as "round_start_time",
      TIMESTAMP 'epoch' + (r."roundEndTime") * INTERVAL '1 second' as "round_end_time",
      r."chainId" AS "chain_id",
      r."id" AS "id",
      r."roundId" AS "roundId",
      (r."metadata" #>> ARRAY['quadraticFundingConfig', 'matchingCap'])::boolean AS "has_matching_cap",
      (r."metadata" #>> ARRAY['quadraticFundingConfig', 'matchingCapAmount'])::double precision AS "matching_cap_amount",
      (r."metadata" #>> ARRAY['quadraticFundingConfig', 'matchingFundsAvailable'])::double precision AS "matching_funds_available",
      (r."metadata" #>> ARRAY['quadraticFundingConfig', 'minDonationThreshold'])::boolean AS "has_min_donation_threshold",
      (r."metadata" #>> ARRAY['quadraticFundingConfig', 'minDonationThresholdAmount'])::double precision AS "min_donation_threshold_amount"
    FROM 
        "public"."Round" AS "r"
    WHERE 
       "r"."roundId" = '{round_address}'
),
FilteredProjects as (
    SELECT 
        r."roundId" AS "roundAddress", 
        r."id" AS "roundId",
        r."chain_id",
        p."projectId" AS "projectId", 
        p."status" AS "status", 
        p."project_name" AS "project_name",
        p."votes" AS "votes",
        r."round_name",
        r."min_donation_threshold_amount",
        p."payoutAddress",
        p."applicationId"
    FROM 
        "public"."ApplicationsInRounds" AS "p"
    JOIN
        FilteredRounds AS "r" ON "r"."id" = "p"."roundId"
        AND p."status" = 'APPROVED'
),
Passports AS (
    SELECT
        LOWER("public"."Passport"."userAddress") AS "voter",
        "public"."Passport"."score" AS "score",
        TIMESTAMP 'epoch' + ("public"."Passport"."scoreTimestamp") * INTERVAL '1 second' as "scoreTimestamp",
        TIMESTAMP 'epoch' + ("public"."Passport"."updatedAt") * INTERVAL '1 second' as "updatedAt",
        "public"."Passport"."stamps" AS "stamps"
    FROM
        "public"."Passport"), 
FilteredVotes AS (
    SELECT 
        DISTINCT LOWER(v."voter") AS "voter",
        pp."score",
        p."project_name",
        v."amount" ,
        v."token" ,
        v."amountUSD" ,
        p."round_name",
        p."payoutAddress",
        p."chain_id",
        TIMESTAMP 'epoch' + (v."tx_timestamp") * INTERVAL '1 second' as "tx_timestamp",
        p."applicationId",
        p."roundAddress"
    FROM 
        "public"."Vote" AS "v"
    JOIN 
        FilteredProjects AS "p" ON "p"."roundId" = "v"."roundId"
        AND "p"."projectId" = "v"."projectId"
    LEFT JOIN 
        Passports as "pp" ON "pp"."voter" = LOWER(v."voter")
)
SELECT 
    *
FROM 
    FilteredVotes
 