import os
import sqlite3
import json

def get_cursor(db_path):
    # Open a single connection to the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    return cursor

class SQLExecutor:
    def __init__(self, db_path):
        self.cursor = get_cursor(db_path)

    def query(self, command):
        cursor = self.cursor
        cursor.execute(command)
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        result = {"columns": columns, "data": rows}
        return json.dumps(result, indent=2)


db_path = os.environ.get("FINTRAN_DB", "/data/fintran.db")
executor = SQLExecutor(db_path)

def query(cmd):
    print(executor.query(cmd))

# if __name__ == "__main__":
#     query("select count(*) as count from users_data")

