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