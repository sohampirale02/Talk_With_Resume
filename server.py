import os
import PyPDF2
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import shutil
from pathlib import Path

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Sequence
from langgraph.prebuilt import ToolNode
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

# Initialize FastAPI
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create uploads directory
UPLOAD_DIR = Path("uploaded_resumes")
UPLOAD_DIR.mkdir(exist_ok=True)

# Initialize LangGraph components
memory = MemorySaver()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def read_pdf(file_path: str):
    """Read text from a PDF file. Needs one field called file_path to read pdf from"""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.6,
    api_key=os.getenv('GEMINI_API_KEY')
).bind_tools([read_pdf])

def agent_node(state: AgentState) -> AgentState:
    """Agent node which talks back and forth with HR and helps HR talk with the Resume"""
    print('inside agent_node')
    system_prompt = SystemMessage(
        content="You are AI assistant of HR manager who helps the HR talk with a resume, and uses read_pdf tool to retrieve data from resume. Ask for path of the resume to the HR. After whole process is done and if HR is satisfied and wants to stop just return the string 'end' ONLY"
    )
    response = llm.invoke([system_prompt] + state['messages'])
    print('AI:', response.content)
    return {"messages": [response]}

def decider(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    print("inside decider")
    if isinstance(last_message, AIMessage):
        if not last_message.tool_calls:
            if last_message.content == 'end':
                return "end"
            else:
                return "continue"
        else:
            return "tool_edge"
    else:
        return "continue"

# Build the graph
graph = StateGraph(AgentState)
graph.add_node("agent_node", agent_node)
graph.add_node("tool_node", ToolNode(tools=[read_pdf]))
graph.add_edge(START, "agent_node")
graph.add_edge("tool_node", "agent_node")

graph.add_conditional_edges(
    "agent_node",
    decider,
    {
        "end": END,
        "continue": END,
        "tool_edge": 'tool_node'
    }
)

agent_app = graph.compile(checkpointer=memory)

# Thread ID for single user
THREAD_ID = "user_session_1"
config = {"configurable": {"thread_id": THREAD_ID}}

# Pydantic models
class ChatMessage(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

# API Endpoints

@app.get("/")
async def root():
    return {"message": "Resume Chat API is running"}

@app.get("/resume")
async def list_resumes():
    """List all uploaded resumes"""
    try:
        resumes = []
        for file_path in UPLOAD_DIR.glob("*"):
            if file_path.is_file():
                resumes.append({
                    "filename": file_path.name,
                    "path": str(file_path),
                    "size": file_path.stat().st_size
                })
        return {"resumes": resumes, "count": len(resumes)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/resume")
async def upload_resume(file: UploadFile = File(...)):
    """Upload a resume (PDF file)"""
    try:
        # Validate file type
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        # Save file
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        return {
            "message": "Resume uploaded successfully",
            "filename": file.filename,
            "path": str(file_path)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def chat(chat_message: ChatMessage):
    """Chat endpoint that processes messages through the LangGraph agent"""
    try:
        user_message = chat_message.message
        
        # Create message and state
        messages = [HumanMessage(content=user_message)]
        initial_state = AgentState(messages=messages)
        
        # Invoke the agent
        response = agent_app.invoke(initial_state, config=config)
        
        # Extract the last AI message
        last_message = response['messages'][-1]
        
        if isinstance(last_message, AIMessage):
            ai_response = last_message.content
        else:
            ai_response = "No response from agent"
        
        return ChatResponse(response=ai_response)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3001)