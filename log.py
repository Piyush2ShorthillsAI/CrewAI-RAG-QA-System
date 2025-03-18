import pandas as pd
import os

def log_interaction(query, response, log_file="logs.csv"):
    """Logs the user query and LLM response into a CSV file."""
    log_entry = pd.DataFrame([{"User Query": query, "LLM Response": response}])  # Convert dict to DataFrame

    if os.path.isfile(log_file):
        # Read existing CSV and append new data using concat
        df = pd.read_csv(log_file)
        df = pd.concat([df, log_entry], ignore_index=True)  # Ensure both are DataFrames
    else:
        # Create new CSV file if it doesn't exist
        df = log_entry

    df.to_csv(log_file, index=False)
