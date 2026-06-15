# scripts/check_db.py
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
db_path = ROOT / "data" / "processed" / "fin.db"

print(f"DB exists: {db_path.exists()}")

if db_path.exists():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    print("Tables:", cursor.fetchall())
    cursor.execute("SELECT AVG(net_profit) FROM quarterly_pl")
    print("Avg net_profit:", cursor.fetchone())
    conn.close()
else:
    print("Run: python scripts/load_sql_db.py")