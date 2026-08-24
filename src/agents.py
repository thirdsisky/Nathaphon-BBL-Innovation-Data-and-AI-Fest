from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from src.retrieval_tool import search_knowledge_base

_llm = None

def get_llm() -> ChatOpenAI:
    """Built on first use, not at import time, otherwise importing this
    module would require the API key to already be in the environment, which
    breaks if the import happens before load_dotenv()."""
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(model="gpt-5-nano", reasoning_effort="medium")
    return _llm

# Agent State
class AgentState(TypedDict):
    user_query: str
    retrieved_snippets: str
    final_report: str

#Agent Definitions
def data_retriever_agent(state: AgentState) -> dict:
    """Agent 1: Runs the retrieval tool and returns the raw text chunks."""
    query = state["user_query"]
    snippets = search_knowledge_base.invoke({"query": query})
    return {"retrieved_snippets": snippets}


def report_generator_agent(state: AgentState) -> dict:
    """Agent 2: Synthesizes the snippets into a clean, well-formatted answer."""
    query = state["user_query"]
    snippets = state["retrieved_snippets"]

    system_prompt = (
        "You are an expert Report Generator. You receive raw context snippets "
        "and a user query. Write an accurate, well-structured answer using only "
        "the provided snippets. Answer the question itself first, then give the "
        "supporting detail the snippets contain -- be thorough rather than "
        "terse, but never state the same fact twice. Write the answer as plain "
        "prose and bullet points, with no section headings. "
        "Never add facts from your own general knowledge. If the snippets lack "
        "the answer, reply with a single sentence saying the knowledge base does "
        "not cover it, and nothing else. "
        "Do not offer to follow up or ask the user questions: your answer is "
        "printed as-is and nobody can reply to it."
    )

    user_prompt = f"""User Query: {query}

Retrieved Information:
{snippets}

Generate a clear, structured final answer:"""

    response = get_llm().invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )

    return {"final_report": response.content}

#Graph Orchestration
def build_agent_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("data_retriever", data_retriever_agent)
    workflow.add_node("report_generator", report_generator_agent)

    workflow.add_edge(START, "data_retriever")
    workflow.add_edge("data_retriever", "report_generator")
    workflow.add_edge("report_generator", END)

    return workflow.compile()
