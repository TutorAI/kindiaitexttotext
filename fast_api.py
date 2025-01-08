from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain import LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from config import GROQ_API_KEY

# Set up your Groq API key (replace this with actual key fetching logic)

# Define the FastAPI app
app = FastAPI()

# Load the LLaMA model using Groq's API
model = ChatGroq(temperature=0, model_name="llama-3.1-70b-versatile",groq_api_key=GROQ_API_KEY)

# Define the prompt template
prompt = """
You are an expert in educational question answering with a focus on maths, physics, chemistry, biology, and general knowledge. When responding to questions, provide accurate and detailed explanations, avoiding unnecessary information. Your answers should be concise, easy to understand, and structured logically.

The conversation so far:
{history}

Please answer the following question in detail: {question}

Please avoid repetition.
"""

template = PromptTemplate(template=prompt, input_variables=["history", "question"])

# Initialize memory for the conversation
memory = ConversationBufferWindowMemory(k=1)

# Create the chain with the model, prompt, and memory
chain = LLMChain(llm=model, prompt=template, memory=memory, verbose=False)

# Pydantic model to handle the input data
class QuestionRequest(BaseModel):
    question: str

# API route to handle question-answering
@app.post("/ask-question/")
async def ask_question(request: QuestionRequest):
    try:
        # Run the question through the chain
        result = chain.run(request.question)
        return {"answer": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the app (for local testing)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
