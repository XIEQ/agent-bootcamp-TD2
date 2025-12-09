import sqlite3
from typing import Tuple, Union

def create_test_cursor() -> Tuple[sqlite3.Connection, sqlite3.Cursor]:
    """
    Creates an in-memory SQLite database connection and cursor, 
    and initializes a 3-row 'test_data' table.

    Returns:
        A tuple containing the sqlite3 Connection and Cursor objects.
    """
    # 1. Create an in-memory SQLite database
    # Passing ':memory:' creates a temporary database in RAM
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()

    # 2. Define the table creation SQL
    create_table_sql = """
    CREATE TABLE test_data (
        name TEXT NOT NULL,
        value INTEGER NOT NULL
    );
    """
    
    # Execute the table creation
    cursor.execute(create_table_sql)

    # 3. Define the data to insert (3 rows)
    test_rows = [
        ('Alice', 10),
        ('Bob', 20),
        ('Charlie', 30)
    ]
    
    # Define the insertion SQL
    insert_sql = "INSERT INTO test_data (name, value) VALUES (?, ?)"

    # Execute the insertion for all 3 rows
    cursor.executemany(insert_sql, test_rows)
    
    # Commit the changes to the database
    conn.commit()
    
    # The cursor and connection are now ready for testing
    return conn, cursor

# --- Example Usage ---

if __name__ == "__main__":
    conn, cursor = create_test_cursor()
    
    print("✅ Test database created in memory.")
    
    try:
        # Example test: Select all data from the table
        print("\n🔎 Executing test query:")
        cursor.execute("SELECT name, value FROM test_data")
        
        # Fetch and print the results
        results = cursor.fetchall()
        
        print(f"Table 'test_data' contains {len(results)} rows:")
        for row in results:
            print(f"  Name: {row[0]}, Value: {row[1]}")
            
    except sqlite3.Error as e:
        print(f"An error occurred during testing: {e}")
        
    finally:
        # 4. Clean up: Close the connection
        # This automatically deletes the in-memory database
        conn.close()
        print("\n🗑️ Connection closed. In-memory database deleted.")