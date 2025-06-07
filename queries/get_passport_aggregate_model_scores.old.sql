SELECT
    lower(address) as address, 
    data_human_probability as "rawScore", 
    updated_at
FROM
    public.passport_model_scores
WHERE 
    model = 'aggregate_model'
    AND lower(address) IN %(addresses)s