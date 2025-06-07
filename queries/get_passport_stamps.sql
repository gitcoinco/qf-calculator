SELECT
  "public"."Passport"."userAddress" AS "address",
  "public"."Passport"."score"::numeric AS "rawScore",
  "public"."Passport"."scoreTimestamp"::timestamptz AS "scoreTimestamp",
  "public"."Passport"."updatedAt"::timestamptz AS "updatedAt",
  "public"."Passport"."stamps" AS "stamps"
FROM
  "public"."Passport"
WHERE
  "public"."Passport"."userAddress" IN %(addresses)s
