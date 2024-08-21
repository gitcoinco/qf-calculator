SELECT 
    a."id",
    (a."metadata" #>> '{application, project, title}')::text AS "project_name",
    (a."metadata" #>> '{application, recipient}')::text AS "recipient_address",
    a."chain_id",
    a."round_id",
    a."project_id",
    a."status",
    a."total_donations_count",
    a."total_amount_donated_in_usd",
    a."unique_donors_count"
FROM 
    "chain_data_75"."applications" AS a
WHERE 
    a.round_id = %(round_id)s AND
    a.chain_id = %(chain_id)s
    AND a."status" = 'APPROVED' ;
