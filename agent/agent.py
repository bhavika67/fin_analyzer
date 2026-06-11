# agent/agent.py
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from loguru import logger
from config import get_settings
from vectorstore.store import VectorStore


class FinancialAgent:
    """ReAct agent using LangGraph (LangChain 1.x compatible)."""

    def __init__(self, vector_store: VectorStore):
        settings = get_settings()
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=0,
        )
        self.vs    = vector_store
        self.agent = self._build_agent()

    # ── Public ────────────────────────────────────────────────────────────────

    def ask(self, question: str) -> dict:
        logger.info(f"Question: {question}")
        try:
            result = self.agent.invoke({
                "messages": [HumanMessage(content=question)]
            })
            # Last message is the final answer
            answer = result["messages"][-1].content
            return {"answer": answer, "error": None}
        except Exception as e:
            logger.error(f"Agent error: {e}")
            return {"answer": None, "error": str(e)}

    # ── Agent builder ─────────────────────────────────────────────────────────

    def _build_agent(self):
        vs = self.vs   # capture for closures

        @tool
        def search_financial_docs(query: str) -> str:
            """Search financial documents for revenue, profit, stock prices,
            company profiles, margins, costs, or any financial metric."""
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
            Use when synthesizing information from several sources."""
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

        system_prompt = """You are a financial analysis assistant with access to a 
knowledge base of financial documents including stock prices, income statements, 
company profiles, and annual reports for AAPL, GOOGL, MSFT, TSLA, INFY, and TCS.

Always use the tools to retrieve information before answering.
Always cite which source document your answer comes from.
Never make up numbers — only use what the tools return."""

        return create_react_agent(
            self.llm,
            tools=[search_financial_docs, summarize_financial_topic],
            prompt=system_prompt,
        )