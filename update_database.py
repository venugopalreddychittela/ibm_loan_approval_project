import sqlite3

DB_PATH = "loan_app.db"

connection = sqlite3.connect(DB_PATH)
cursor = connection.cursor()

try:
    cursor.execute(
        "ALTER TABLE applications ADD COLUMN credit_score INTEGER"
    )

    connection.commit()

    print("Credit score column added successfully.")

except sqlite3.OperationalError as error:

    print("Database update:", error)

finally:

    connection.close()