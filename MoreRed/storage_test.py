import os

import sqlite3
print("nihao")

def print_current_directory_info():
    cwd = os.getcwd()
    print(f"\n📁 Current working directory: {cwd}")
    print("📂 Contents of current directory:")
    for entry in os.listdir(cwd):
        print(f"  - {entry}")
    print()

def print_data(data_dir):
    if not os.path.exists(data_dir):
        print(f"[!] Directory not found: {data_dir}")
        return

    print(f"\n🔍 Reading data from: {data_dir}")
    for file in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file)
        if os.path.isfile(file_path):
            print(f"  - {file}")

# MAIN
print_current_directory_info()

# Absolute path inside container to the mounted dataset
dataset_path = "/input-data/energy_U0"

print("--- CONTENTS OF QM9 DATASET DIR ---")
print_data(dataset_path)

db_path = "/input-data/energy_U0/qm9.db"
print("File exists?", os.path.exists(db_path))

# Try connecting as SQLite
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# List tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
print("Tables:", cursor.fetchall())

# Look at one system (id=1)
cursor.execute("SELECT * FROM number_key_values WHERE id=1;")
print("Number values:", cursor.fetchall())

cursor.execute("SELECT * FROM text_key_values WHERE id=1;")
print("Text values:", cursor.fetchall())

conn.close()


