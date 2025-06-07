SELECT
  "public"."Passport"."userAddress" AS "address",
  "public"."Passport"."score" AS "rawScore",
  to_timestamp("public"."Passport"."scoreTimestamp") AS "scoreTimestamp",
  to_timestamp("public"."Passport"."updatedAt") AS "updatedAt",
  "public"."Passport"."stamps" AS "stamps"
FROM
  "public"."Passport"
WHERE
  "public"."Passport"."userAddress" IN %(addresses)s
