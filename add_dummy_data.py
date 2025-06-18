import csv
import os
from datetime import datetime, timedelta

DATA_DIR = 'data'

# Dummy data
income_rows = [
    ['Salary', '50000', '2025-06-01'],
    ['Freelance', '12000', '2025-06-10'],
    ['Gift', '3000', '2025-06-15'],
]

expense_rows = [
    ['Groceries', '8000', '2025-06-05'],
    ['Rent', '20000', '2025-06-01'],
    ['Utilities', '3500', '2025-06-08'],
]

goal_rows = [
    ['Vacation', '50000', '10000', '2025-12-31'],
    ['New Laptop', '70000', '20000', '2025-09-30'],
    ['Emergency Fund', '100000', '25000', '2026-06-01'],
]

def add_dummy_to_csv(filename, rows):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        print(f"{filename} not found, skipping.")
        return
    with open(path, 'r', newline='', encoding='utf-8') as f:
        lines = list(csv.reader(f))
    # Only add if only header is present
    if len(lines) <= 1:
        with open(path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(rows)
        print(f"Added dummy data to {filename}")
    else:
        print(f"{filename} already has data, skipping.")

def main():
    add_dummy_to_csv('income.csv', income_rows)
    add_dummy_to_csv('expenses.csv', expense_rows)
    add_dummy_to_csv('goals.csv', goal_rows)

if __name__ == '__main__':
    main() 