# scripts/ask.py
"""
Interactive Q&A with the financial agent.
Usage: python scripts/ask.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import os
from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=True)

from vectorstore.store import VectorStore
from agent.agent import FinancialAgent

def main():
    print("Loading vector store...")
    store = VectorStore()
    store.load()
    print(f"Loaded {store.total} vectors\n")

    agent = FinancialAgent(store)

    questions = [
    "What is the average net_profit across all quarters in quarterly_pl?",
    "Which quarter had the highest revenue in quarterly_pl?",
    "Compare AAPL and MSFT net income from their financials tables",
    "What was the total revenue for FY 2023?",  # the one that failed RAGAS
    "What were the key risks mentioned in the annual report?",
]

    for q in questions:
        print("\n" + "="*60)
        print(f"Q: {q}")
        print("="*60)
        result = agent.ask(q)
        if result["error"]:
            print(f"ERROR: {result['error']}")
        else:
            print(f"\nFINAL ANSWER:\n{result['answer']}")

if __name__ == "__main__":
    main()