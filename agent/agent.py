# agent/agent.py
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from loguru import logger
from pathlib import Path
import sqlite3
import pandas as pd
import uuid
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from config import get_settings
from vectorstore.store import VectorStore

ROOT    = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "processed" / "fin.db"


class FinancialAgent:
    """ReAct agent using LangGraph — RAG search + SQL query + chart tools."""

    def __init__(self, vector_store: VectorStore):
        settings = get_settings()
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=0,
        )
        self.vs    = vector_store
        self.agent = self._build_agent()

    def ask(self, question: str) -> dict:
        logger.info(f"Question: {question}")
        try:
            result = self.agent.invoke({
                "messages": [HumanMessage(content=question)]
            })
            answer = result["messages"][-1].content
            return {"answer": answer, "error": None}
        except Exception as e:
            logger.error(f"Agent error: {e}")
            return {"answer": None, "error": str(e)}

    def _build_agent(self):
        vs = self.vs

        @tool
        def search_financial_docs(query: str) -> str:
            """Search financial documents (PDFs, profiles, annual reports) for
            qualitative information — risks, outlook, commentary, company descriptions.
            Use for narrative/text-based questions, NOT for numeric calculations."""
            results = vs.search(query, top_k=4)
            if not results:
                return "No relevant documents found for this query."
            parts = []
            for i, r in enumerate(results, 1):
                source = r.get("filename", r.get("source", "unknown"))
                score  = r.get("score", 0)
                text   = r.get("text", "")[:800]
                parts.append(f"[{i}] Source: {source} (relevance: {score:.3f})\n{text}")
            return "\n\n---\n\n".join(parts)

        @tool
        def summarize_financial_topic(topic: str) -> str:
            """Summarize findings across multiple documents on a financial topic.
            Use when synthesizing qualitative information from several sources."""
            results = vs.search(topic, top_k=6)
            if not results:
                return "No relevant documents found to summarize."
            context = "\n\n".join([r.get("text", "")[:500] for r in results])
            response = self.llm.invoke([
                {"role": "system", "content": (
                    "You are a financial analyst. Summarize the following excerpts "
                    "in 3-5 bullet points. Cite the source filename for each point."
                )},
                {"role": "user", "content": f"Topic: {topic}\n\nExcerpts:\n{context}"},
            ])
            return response.content

        @tool
        def query_financial_database(sql_query: str) -> str:
            """Run a read-only SQL query against the financial database for
            precise numeric calculations — averages, sums, comparisons, growth rates,
            specific quarter values. Use this for ANY question involving numbers,
            calculations, or comparisons across periods/companies.

            Available tables and columns:
            - quarterly_pl(quarter, revenue, cogs, opex, ebitda, tax, net_profit, net_margin)
            - segment_revenue(month, segment, revenue, customers, churn_rate)
            - cost_headcount(month, headcount, salary_cost, infra_cost, marketing, rd_spend, total_cost)
            - aapl_financials(period, revenue, gross_profit, operating_income, net_income, gross_margin, net_margin)
            - googl_financials(period, revenue, gross_profit, operating_income, net_income, gross_margin, net_margin)
            - msft_financials(period, revenue, gross_profit, operating_income, net_income, gross_margin, net_margin)
            - tsla_financials(period, revenue, gross_profit, operating_income, net_income, gross_margin, net_margin)
            - infy_financials(period, revenue, gross_profit, operating_income, net_income, gross_margin, net_margin)
            - tcs_financials(period, revenue, gross_profit, operating_income, net_income, gross_margin, net_margin)

            Only SELECT statements are allowed. Always use SQLite syntax.
            Example: SELECT AVG(revenue) FROM quarterly_pl
            Example: SELECT quarter, net_profit FROM quarterly_pl ORDER BY net_profit DESC LIMIT 1
            """
            sql_lower = sql_query.strip().lower()
            if not sql_lower.startswith("select"):
                return "Error: Only SELECT queries are allowed."
            if any(kw in sql_lower for kw in ["drop", "delete", "update", "insert", "alter"]):
                return "Error: Query contains forbidden keywords."

            if not DB_PATH.exists():
                return "Error: Database not found. Run scripts/load_sql_db.py first."

            try:
                conn = sqlite3.connect(DB_PATH)
                df = pd.read_sql_query(sql_query, conn)
                conn.close()
                if df.empty:
                    return "Query returned no results."
                return df.to_string(index=False)
            except Exception as e:
                return f"SQL error: {e}. Check table/column names in the schema description."

        @tool
        def generate_chart(request: str) -> str:
            """Generate a chart image from the financial database.
            Input format: 'chart_type|table|x_column|y_column'

            chart_type: line, bar, or scatter
            table: any table from query_financial_database's schema
                   (quarterly_pl, segment_revenue, cost_headcount,
                    aapl_financials, googl_financials, msft_financials,
                    tsla_financials, infy_financials, tcs_financials)

            Examples:
            - 'line|quarterly_pl|quarter|revenue'
            - 'bar|aapl_financials|period|net_income'
            - 'line|cost_headcount|month|total_cost'

            Use this when the user asks to 'show', 'plot', 'visualize',
            or 'chart' any financial trend.
            """
            try:
                parts = [p.strip() for p in request.split('|')]
                if len(parts) != 4:
                    return "Invalid format. Use: chart_type|table|x_column|y_column"

                chart_type, table, x_col, y_col = parts

                if not DB_PATH.exists():
                    return "Error: Database not found. Run scripts/load_sql_db.py first."

                conn = sqlite3.connect(DB_PATH)
                try:
                    df = pd.read_sql_query(f"SELECT {x_col}, {y_col} FROM {table}", conn)
                finally:
                    conn.close()

                if df.empty:
                    return f"No data found in table '{table}'."

                # Style — dark theme matching the UI
                plt.style.use('dark_background')
                fig, ax = plt.subplots(figsize=(9, 4))
                fig.patch.set_facecolor('#1a1d27')
                ax.set_facecolor('#1a1d27')

                if chart_type == 'line':
                    ax.plot(df[x_col].astype(str), df[y_col],
                            color='#4f8ef7', linewidth=2, marker='o', markersize=4)
                    ax.fill_between(range(len(df)), df[y_col], alpha=0.1, color='#4f8ef7')
                elif chart_type == 'bar':
                    colors = ['#34d17a' if v >= 0 else '#f05b5b' for v in df[y_col]]
                    ax.bar(df[x_col].astype(str), df[y_col], color=colors, width=0.6)
                elif chart_type == 'scatter':
                    ax.scatter(df[x_col], df[y_col], color='#4f8ef7', alpha=0.7, s=40)
                else:
                    plt.close()
                    return f"Unknown chart_type '{chart_type}'. Use line, bar, or scatter."

                ax.set_xlabel(x_col, color='#7b8099', fontsize=10)
                ax.set_ylabel(y_col, color='#7b8099', fontsize=10)
                ax.set_title(f'{y_col} by {x_col} ({table})', color='#e8eaf2', fontsize=12, pad=12)
                ax.tick_params(colors='#7b8099')
                for spine in ax.spines.values():
                    spine.set_color('#2a2e45')

                if len(df) > 10:
                    step = max(1, len(df) // 10)
                    ax.set_xticks(range(0, len(df), step))
                plt.xticks(rotation=45, ha='right', fontsize=8)
                plt.tight_layout()

                # Save to ui/charts/
                out_dir = ROOT / "ui" / "charts"
                out_dir.mkdir(parents=True, exist_ok=True)
                chart_id  = str(uuid.uuid4())[:8]
                out_path  = out_dir / f"chart_{chart_id}.png"
                plt.savefig(out_path, dpi=110, bbox_inches='tight', facecolor='#1a1d27')
                plt.close()

                return f"CHART_GENERATED:{out_path}|{y_col} by {x_col} ({table})"

            except Exception as e:
                return f"Chart error: {str(e)}. Check table/column names match the SQL schema."

        # agent/agent.py — update system_prompt, add this paragraph at the end:

        system_prompt = """You are a financial analysis assistant with access to:
1. A document search tool for qualitative info (risks, outlook, company profiles, annual reports)
2. A SQL database tool for precise numeric calculations (revenue, profit, margins, growth, comparisons)
3. A chart generation tool for visualizing financial trends

For NUMERIC questions (averages, totals, comparisons, "highest", "growth rate", specific values) —
ALWAYS use query_financial_database first.

For QUALITATIVE questions (risks, strategy, outlook, "why did X happen") —
use search_financial_docs or summarize_financial_topic.

For requests to 'show', 'plot', 'visualize', or 'chart' data —
use generate_chart with the correct table and column names from the SQL schema below.

IMPORTANT: When generate_chart returns a string starting with "CHART_GENERATED:",
output that EXACT string as your final answer, verbatim, with NO markdown formatting,
NO image syntax, NO additional text before or after it. Do not wrap it in ![]() or
any other markdown. Just output the raw "CHART_GENERATED:path|title" string alone.

Some questions need multiple tools — use SQL for the numbers and document search for the "why".

Always cite your source (table name or document filename).
Never make up numbers — only use what the tools return."""

        return create_react_agent(
            self.llm,
            tools=[search_financial_docs, summarize_financial_topic,
                   query_financial_database, generate_chart],
            prompt=system_prompt,
        )