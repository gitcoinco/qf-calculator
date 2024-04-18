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
        r."min_donation_threshold_amount"
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
        ARRAY_AGG(p."project_name" ORDER BY p."project_name")  AS "project_names",
        ARRAY_AGG(v."amount" ORDER BY p."project_name") AS "amounts",
        ARRAY_AGG(v."token" ORDER BY p."project_name")  AS "tokens",
        ARRAY_AGG(v."amountUSD" ORDER BY p."project_name")  AS "amountsUSD",
        ARRAY_AGG(p."round_name" ORDER BY p."project_name")  AS "round_names",
        AVG(v."amountUSD")  AS "average_amountUSD",
        SUM(v."amountUSD")  AS "total_amountUSD",
        COUNT(v."amountUSD") AS "total_votes"
    FROM 
        "public"."Vote" AS "v"
    JOIN 
        FilteredProjects AS "p" ON "p"."roundId" = "v"."roundId"
        AND "p"."projectId" = "v"."projectId"
    LEFT JOIN 
        Passports as "pp" ON "pp"."voter" = LOWER(v."voter")
    GROUP BY 
        v."voter",
        pp."score"
    ORDER BY
    "voter", "score"
)

SELECT 
    *
FROM 
    FilteredVotes




