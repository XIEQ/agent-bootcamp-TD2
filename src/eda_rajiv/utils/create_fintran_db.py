import sqlite3
import pandas as pd
from pathlib import Path

import json


def load_json(file_path):
    with open(file_path, "r") as f:    # except sqlite3.Error as e:
    #     print(f"Database error: {{e}}")
    #     if conn:
    #         conn.rollback() # Rollback changes on error
    # finally:
    #     if conn:
    #         conn.close() # Close the single connection when done


        return json.load(f)


def cards_data_creator(file_path):
    df = pd.read_csv(file_path)

    df[['expires_month', 'expires_year']] = (
        df['expires']
        .str.split('/', expand=True)
    )

    df['expires_month'] = df['expires_month'].astype(int)
    df['expires_year'] = df['expires_year'].astype(int)

    df.drop('expires', axis=1, inplace=True)

    df[['acct_open_month', 'acct_open_year']] = (
        df['acct_open_date']
        .str.split('/', expand=True)
    )

    df['acct_open_month'] = df['acct_open_month'].astype(int)
    df['acct_open_year'] = df['acct_open_year'].astype(int)

    df.drop('acct_open_date', axis=1, inplace=True)

    # df['credit_limit'] = df['credit_limit'].str.replace('$', '', regex=False)

    # df['credit_limit'] = df['credit_limit'].astype(float)
    df = dollar_to_float(df, "credit_limit")
    return df

def dollar_to_float(df, column):
    df[column] = df[column].str.replace('$', '', regex=False)

    df[column] = df[column].astype(float)
    return df

def clean_zip(df, column):
    import pandas as pd

    # 1. Convert the column to string type first, to safely handle the .0
    df[column] = df[column].astype(str)

    # 2. Use a regular expression to target and remove only the '.0' at the END of the string.
    # r'\.0$' means: a literal dot (\.), followed by a zero (0), at the end of the string ($).
    df[column] = df[column].str.replace(r'\.0$', '', regex=True)
    return df

def users_data_creator(file_path):
    df = pd.read_csv(file_path)

    df = dollar_to_float(df, "per_capita_income")
    df = dollar_to_float(df, "yearly_income")
    df = dollar_to_float(df, "total_debt")
    return df

def transactions_data_creator(file_path):
    df = pd.read_csv(file_path)

    df = dollar_to_float(df, "amount")
    df = clean_zip(df, "zip")
    return df

def mcc_creator(file_path):
    data = load_json(file_path)
    # TODO: dedupe

    # Convert to DataFrame
    return pd.DataFrame(
        [{"id": key, "description": value} for key, value in data.items()]
    ).reset_index(drop=True)


def train_fraud_file_creator(file_path):
    data = load_json(file_path)
    data = data["target"]
    return pd.DataFrame(
        [{"id": key, "label": value} for key, value in data.items()]
    ).reset_index(drop=True)


def data_file_info(file_name, creator_func, sql_create_cmd):
    return {
        "file_name": file_name,
        "creator": creator_func,
        "sql_create_cmd": sql_create_cmd,
    }



CARDS_DATA_TABLE = """
CREATE TABLE cards_data (
    id INTEGER PRIMARY KEY,
    client_id INTEGER,
    card_brand TEXT,
    card_type TEXT,
    card_number TEXT UNIQUE,
    expires_month INT,
    expires_year INT,
    cvv INTEGER,
    has_chip TEXT,
    num_cards_issued INTEGER,
    credit_limit FLOAT,  -- Stored as TEXT due to '$' sign
    acct_open_month INTEGER,
    acct_open_year INTEGER,
    year_pin_last_changed INTEGER,
    card_on_dark_web TEXT,
    FOREIGN KEY (client_id) REFERENCES users_data(client_id)
);
"""



MCC_CODES_TABLE = """
CREATE TABLE mcc_codes (
id TEXT PRIMARY KEY,
description TEXT
);
"""
TRAIN_FRAUD_LABELS_TABLE = """
CREATE TABLE train_fraud_labels (
id TEXT PRIMARY KEY,
label TEXT
);
"""
USERS_DATA_TABLE = """
CREATE TABLE users_data (
    id INTEGER PRIMARY KEY,
    current_age INTEGER,
    retirement_age INTEGER,
    birth_year INTEGER,
    birth_month INTEGER,
    gender TEXT,
    address TEXT,
    latitude REAL,
    longitude REAL,
    per_capita_income FLOAT,
    yearly_income FLOAT,
    total_debt FLOAT,
    credit_score INTEGER,
    num_credit_cards INTEGER
);
"""
TRANSACTIONS_DATA_TABLE = """
CREATE TABLE transactions_data (
    id INTEGER PRIMARY KEY,
    date DATETIME,
    client_id INTEGER,
    card_id INTEGER,
    amount FLOAT,
    use_chip TEXT,
    merchant_id INTEGER,
    merchant_city TEXT,
    merchant_state TEXT,
    zip TEXT,
    mcc TEXT,
    errors TEXT,
    FOREIGN KEY (client_id) REFERENCES users_data(client_id),
    FOREIGN KEY (card_id) REFERENCES cards_data(id),
    FOREIGN KEY (mcc) REFERENCES mcc_codes(id)
);
"""

DATA_FILES = {
    # "cards_data": data_file_info("cards_data.csv", pd.read_csv, CARDS_DATA_TABLE),
    "cards_data": data_file_info("cards_data.csv", cards_data_creator, CARDS_DATA_TABLE),
    "mcc_codes": data_file_info("mcc_codes.json", mcc_creator, MCC_CODES_TABLE),
    "train_fraud_labels": data_file_info(
        "train_fraud_labels.json", train_fraud_file_creator, TRAIN_FRAUD_LABELS_TABLE
    ),
    "users_data": data_file_info("users_data.csv", users_data_creator, USERS_DATA_TABLE),
    "transactions_data": data_file_info(
        "transactions_data.csv", transactions_data_creator, TRANSACTIONS_DATA_TABLE
    ),
}


def db_get_connection(db_path):
    print(f"Db Path Raj: {db_path}")
    return sqlite3.connect(db_path)


def db_create_schema(conn):
    # conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Drop existing tables if they exist
    for table_name in DATA_FILES:
        cursor.execute(f"DROP TABLE IF EXISTS {table_name}")

    for _, table_info in DATA_FILES.items():
        cursor.execute(table_info["sql_create_cmd"])
    conn.commit()


def db_load_data(conn, data_path, tables=None):
    """Load CSV data from data/fintran into SQLite database."""
    data_dir = Path(data_path)

    if not data_dir.exists():
        print(f"Data directory not found: {data_path}")
        return

    for table_name, data_info in DATA_FILES.items():
        if not tables or table_name in tables:
            print(f"Loading {table_name}")
            df_func = data_info["creator"]
            file_path = data_dir / data_info["file_name"]
            df = df_func(file_path)
            df.to_sql(table_name, conn, if_exists="append", index=False)
    conn.commit()

    print("Data loading complete!")


def db_run_query(conn, query):
    """Execute SQL query using conn and return results as a pandas DataFrame."""
    if not query:
        raise ValueError("query must be a non-empty SQL string")

    cursor = conn.cursor()
    cursor.execute(query)
    cols = [col[0] for col in cursor.description] if cursor.description else []
    rows = cursor.fetchall()
    cursor.close()

    if cols:
        return pd.DataFrame(rows, columns=cols).reset_index(drop=True)
    # Non-SELECT query (no columns) -> empty DataFrame
    return pd.DataFrame()


def verify_database(db_path):
    """Verify the database was created and populated correctly"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM customers")
    customer_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM transactions")
    transaction_count = cursor.fetchone()[0]

    print(f"\n✓ Database created: {db_path}")
    print(f"✓ Customers: {customer_count}")
    print(f"✓ Transactions: {transaction_count}")

    conn.close()


# if __name__ == "__main__":
#     # Create database and tables
#     conn = create_database()
#     print("Database and tables created!")

#     # Load data
#     load_data_to_db(conn)

#     # Verify
#     verify_database()

#     conn.close()
