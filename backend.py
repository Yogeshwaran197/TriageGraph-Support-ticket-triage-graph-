
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import MessagesPlaceholder
from langgraph.graph import StateGraph
from langgraph.types import Command, interrupt
from langgraph.checkpoint.postgres import PostgresSaver
from typing import TypedDict, Dict, Literal
from pydantic import BaseModel, Field
from psycopg import connect
from psycopg.rows import dict_row

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("Invalid Groq API key")


class ticket(BaseModel):
    ticket_id: int = Field(..., description="Unique numeric ID of the support ticket, e.g. 1042. Used to look up the ticket record.")
    ticket_text : str = Field(description="The customer's original message describing their problem. Pass it verbatim, without summarizing or editing.")

class category(BaseModel):
    category: Literal["billing", "technical", "account", "shipping", "general"] = Field(
    description=(
        "Category assigned to the ticket after classification. One of: "
        "'billing' (refunds, charges, invoices, payment failures), "
        "'technical' (bugs, errors, crashes, app not working), "
        "'account' (login, password reset, profile or email changes), "
        "'shipping' (delivery delays, tracking, lost or damaged orders), "
        "'general' (anything that doesn't fit the above, or questions about policies)."
    )
)

class AgentState(TypedDict):
    ticket :ticket
    category : category
    retrived_context : str
    draft : str
    approval : Literal["approve", "edit", "reject"]
    retry_count: int


llm = ChatGroq(
    model = "openai/gpt-oss-20b",
    groq_api_key = os.getenv("GROQ_API_KEY")
)




