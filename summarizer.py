
import os 
import pandas as pd
from tabulate import tabulate


df = pd.read_excel('/06_2024_class.xlsx')

df["Debit CHF"] = pd.to_numeric(df["Debit CHF"], errors="coerce")
summary = df.groupby("Category")["Debit CHF"].sum()
summary_df = summary.reset_index()

month = "06-2024"

out = f"/{month}_expenses.txt"
os.makedirs(os.path.dirname(out), exist_ok=True)

with open(out, "w") as f:
    f.write(tabulate(summary_df, headers="keys", tablefmt="psql", showindex=False))

print("summary saved!")