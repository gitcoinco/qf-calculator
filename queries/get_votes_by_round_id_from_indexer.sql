SELECT
    (a."metadata" #>> '{application, project, title}')::text AS "project_name",
    d.id,
    d.chain_id,
    d.round_id,
    d.application_id,
    d.donor_address as "voter",
    d.recipient_address,
    d.project_id,
    d.transaction_hash,
    d.block_number,
    d.token_address,
    d.amount,
    d.amount_in_usd as "amountUSD",
    d.amount_in_round_match_token
FROM
    "chain_data_63".donations d
LEFT JOIN "chain_data_63".applications a ON a.round_id = d.round_id AND a.id = d.application_id
AND a.status = 'APPROVED'
WHERE
    d.round_id = %(round_id)s AND
    d.chain_id = %(chain_id)s;
