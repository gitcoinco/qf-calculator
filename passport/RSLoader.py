import json
from sqlalchemy import create_engine
import pandas as pd

# Initialize empty list for results
results = []

# Load and parse JSONL file
try:
    with open("registry_score.jsonl", 'r', encoding='utf-8') as file:
        for i, line in enumerate(file):
            line = line.strip()
            if line:  # Skip empty lines
                try:
                    parsed_json = json.loads(line)
                    
                    print("Parsed Line:",i)

                    # Transform the data to match database schema
                    if parsed_json.get("last_score_timestamp"):
                        transformed_record = {
                            "userAddress": parsed_json["passport"]["address"],
                            "score": parsed_json.get('evidence', {}).get('rawScore') if parsed_json.get('evidence') is not None else 0,
                            "evidence": json.dumps(parsed_json.get('evidence', {})),
                            "scoreTimestamp": parsed_json.get("last_score_timestamp"),
                            "updatedAt": parsed_json.get("last_score_timestamp"),
                            "stamps": json.dumps(parsed_json["stamps"])
                        }
                        results.append(transformed_record)
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON on line {i+1}: {e}")
                    continue
except FileNotFoundError:
    print(f"Error: JSONL file not found: registry_score.jsonl")
    exit(1)

# Convert results to DataFrame
df = pd.DataFrame(results)

# Insert data into database
try:
    # Create SQLAlchemy engine
    engine = create_engine('postgresql://your_username:your_password@localhost:5432/indexer')
    
    # Insert data into database
    df.to_sql('Passport', engine, if_exists='append', index=False)
    print(f"Successfully loaded {len(df)} records to database")
except Exception as e:
    print(f"Error inserting data into database: {e}")

print(f"Processed {len(results)} records")
