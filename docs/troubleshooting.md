# Troubleshooting Guide

## Common Issues

### 1. Passport Scoring Gaps
Sometimes passport scores will only fill for 90% of the users. 
- Best solution: Contact the passport team directly to get the missing scores
- Who to contact: Nadjib, Stefi, or Jeremy

### 2. Matching Overflow Errors 
Be very careful when adjusting the way the matching is calculated.
- If unsure, ask Joel for help
- The matching distribution CSV gets posted onchain - small errors can cause large payout issues
- Important: Matching amounts use extremely high precision (usually 10^18 decimals)
- Warning: Python's standard floating point precision is insufficient
- Current configuration should prevent overflows
- When changing matching logic, test thoroughly

### 3. Missing Data
Sometimes the indexer will be reindexing and data will be missing.
- This is usually temporary and resolves once indexing completes
- Solutions:
  - Wait for indexer to finish reindexing
  - Contact engineering team directly
