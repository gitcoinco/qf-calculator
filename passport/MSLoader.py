import pandas as pd
from sqlalchemy import create_engine
import json

# Read the recovered parquet file
df = pd.read_parquet(path='./model_scores.parquet', engine='pyarrow')

# Parse the JSON data column and extract human_probability
def extract_human_probability(json_str):
    try:
        data = json.loads(json_str)
        return data.get('data', {}).get('human_probability', 0)
    except:
        return 0

# Assuming the columns are in order: [json_data, timestamp, id, model_type, address]
# Adjust column names based on actual column names in your parquet file
df.columns = ['json_data', 'updated_at', 'id', 'model_type', 'address']

# Transform the data to match the expected schema
df = df.assign(
    address=lambda x: x['address'].str.lower(),  # Ensure address is lowercase
    data_human_probability=lambda x: x['json_data'].apply(extract_human_probability),  # Extract human_probability as rawScore
    model='aggregate_model'  # Set model to 'aggregate_model' as expected by the query
)

# Select only the columns needed for the database
df_final = df[['address', 'data_human_probability', 'updated_at', 'model']]

# Create SQLAlchemy engine
engine = create_engine('postgresql://your_username:your_password@localhost:5432/indexer')

# Import to PostgreSQL using SQLAlchemy
df_final.to_sql('passport_model_scores', engine, if_exists='replace', index=False)

print(f"Successfully loaded {len(df_final)} rows to PostgreSQL")