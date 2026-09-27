import os
import sqlite3
from pathlib import Path

from dotenv import load_dotenv
import certifi

load_dotenv()

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver
from tools import tools

Path("data").mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# LLM provider configuration
#
# To switch provider, set LLM_PROVIDER in .env (or the environment):
#     LLM_PROVIDER=openai   -> uses OPENAI_API_KEY / OPENAI_MODEL
#     LLM_PROVIDER=gemini   -> uses GOOGLE_API_KEY / GOOGLE_MODEL
#
# To add/remove models, edit the "models" list of a provider below.
# The first model in the list is shown first in the frontend dropdown.
# ---------------------------------------------------------------------------

def _build_openai_llm(model_name: str):
    # GPT-5.x / GPT-6 models only accept temperature=1. Set it explicitly,
    # otherwise langchain-openai sends its own default of 0.7 and the API rejects it.
    return ChatOpenAI(
        model=model_name,
        temperature=1,
        api_key=os.getenv("OPENAI_API_KEY"),
        # These models reject tool calling on /v1/chat/completions; use /v1/responses.
        use_responses_api=True,
        streaming=True
    )


def _build_gemini_llm(model_name: str):
    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0.3,
        streaming=True
    )


PROVIDERS = {
    "openai": {
        "default_model": os.getenv("OPENAI_MODEL", "gpt-5.6-terra"),
        "models": [
            "gpt-5.6-terra",  # balanced, ≈ gemini-2.5-flash
            "gpt-5.6-sol",    # GPT-5.6 flagship, ≈ gemini-2.5-pro
            "gpt-5.6-luna",   # fastest/cheapest, ≈ gemini-2.5-flash-lite
            "gpt-6-astra",    # top flagship for hardest reasoning/coding
            "gpt-5.5",        # previous-gen fallback
        ],
        "build_llm": _build_openai_llm,
    },
    "gemini": {
        "default_model": os.getenv("GOOGLE_MODEL", "gemini-2.5-flash"),
        "models": [
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-2.5-flash-lite",
            "gemini-1.5-flash",
            "gemini-1.5-pro",
        ],
        "build_llm": _build_gemini_llm,
    },
}

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").strip().lower()

if LLM_PROVIDER not in PROVIDERS:
    raise ValueError(
        f"Unknown LLM_PROVIDER '{LLM_PROVIDER}'. Use one of: {', '.join(PROVIDERS)}"
    )

_PROVIDER = PROVIDERS[LLM_PROVIDER]
ALLOWED_MODELS = _PROVIDER["models"]
DEFAULT_MODEL = _PROVIDER["default_model"]

# Ignore an OPENAI_MODEL / GOOGLE_MODEL value that isn't in the provider's list.
if DEFAULT_MODEL not in ALLOWED_MODELS:
    DEFAULT_MODEL = ALLOWED_MODELS[0]


def get_model_options() -> dict:
    """
    Models for the frontend dropdown, for the active provider.
    """

    return {
        "provider": LLM_PROVIDER,
        "models": ALLOWED_MODELS,
        "default_model": DEFAULT_MODEL,
    }


SYSTEM_PROMPT = """
You are a helpful Agentic AI assistant named MyGPT similar to ChatGPT.

You can:
1. Answer normal questions.
2. Use tools when needed.
3. Search uploaded documents using the RAG tool.
4. Search the web for latest/current information using Tavily Search.
5. Remember important user information using the memory tool.
6. Recall memory when useful.
7. Use calculator for math.

Rules:
- If the user asks about latest news, current events, recent updates, today's information, current prices, current people, current versions, new releases, or anything time-sensitive, use Tavily Search.
- If the user asks about an uploaded document, use search_uploaded_documents.
- If the user asks you to remember something, use remember_this.
- If the user asks about previous preferences or saved facts, use recall_memory.
- Use calculator for math questions.
- When using web search, summarize clearly and mention that the answer is based on web search results.
- Be clear, helpful, and concise.
"""



def normalize_model_name(model_name: str | None) -> str:
    """
    Validate selected model from frontend.
    If model is missing or not allowed, fallback to DEFAULT_MODEL.
    """

    if not model_name:
        return DEFAULT_MODEL

    model_name = model_name.strip()

    if model_name not in ALLOWED_MODELS:
        return DEFAULT_MODEL

    return model_name




def build_agent(model_name: str):
    """
    Build one LangGraph agent for a selected model of the active provider.
    """

    selected_model = normalize_model_name(model_name)

    llm = _PROVIDER["build_llm"](selected_model)

    llm_with_tools = llm.bind_tools(tools)

    def chatbot_node(state: MessagesState):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]

        response = llm_with_tools.invoke(messages)

        return {
            "messages": [response]
        }

    tool_node = ToolNode(tools)

    workflow = StateGraph(MessagesState)

    workflow.add_node("chatbot", chatbot_node)
    workflow.add_node("tools", tool_node)

    workflow.add_edge(START, "chatbot")
    workflow.add_conditional_edges("chatbot", tools_condition)
    workflow.add_edge("tools", "chatbot")

    conn = sqlite3.connect(
        "data/langgraph_checkpoints.sqlite",
        check_same_thread=False
    )

    checkpointer = SqliteSaver(conn)

    return workflow.compile(checkpointer=checkpointer)


_AGENT_CACHE = {}


def get_agent(model_name: str | None = None):
    """
    Return cached LangGraph agent for selected model.
    If not created yet, create it once and reuse it.
    """

    selected_model = normalize_model_name(model_name)

    if selected_model not in _AGENT_CACHE:
        _AGENT_CACHE[selected_model] = build_agent(selected_model)

    return _AGENT_CACHE[selected_model]