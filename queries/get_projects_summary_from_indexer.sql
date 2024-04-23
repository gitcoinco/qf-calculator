SELECT "chain_data_3287eeeb342085_62"."applications"."id" AS "id",
    (
        "chain_data_3287eeeb342085_62"."applications"."metadata"#>>array [ 'application',
    'project',
    'title' ]::text [ ]
    )::text AS "project_name",
    (
        "chain_data_3287eeeb342085_62"."applications"."metadata"#>>array [ 'application',
    'recipient' ]::text [ ]
    )::text AS "recipient_address",
    "chain_data_3287eeeb342085_62"."applications"."chain_id" AS "chain_id",
    "chain_data_3287eeeb342085_62"."applications"."round_id" AS "round_id",
    "chain_data_3287eeeb342085_62"."applications"."project_id" AS "project_id",
    "chain_data_3287eeeb342085_62"."applications"."status" AS "status",
    "chain_data_3287eeeb342085_62"."applications"."total_donations_count" AS "total_donations_count",
    "chain_data_3287eeeb342085_62"."applications"."total_amount_donated_in_usd" AS "total_amount_donated_in_usd",
    "chain_data_3287eeeb342085_62"."applications"."unique_donors_count" AS "unique_donors_count"
FROM "chain_data_3287eeeb342085_62"."applications"
WHERE (
        "chain_data_3287eeeb342085_62"."applications"."round_id" = '{round_address}' AND
        "chain_data_3287eeeb342085_62"."applications"."chain_id" = '{chain_id}' 
    )
    AND (
        "chain_data_3287eeeb342085_62"."applications"."status" = CAST(
            'APPROVED' AS "chain_data_3287eeeb342085_62"."application_status"
        )
    )