import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from backend.vectorstore import get_embeddings, init_chroma
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))

def format_docs(docs):
    """Format retrieved documents into a single context string."""
    return "\n\n".join(doc.page_content for doc in docs)

def get_llm():
    """Initialize Groq LLM."""
    return ChatGroq(
        model="llama-3.3-70b-versatile", 
        temperature=LLM_TEMPERATURE,
        api_key=GROQ_API_KEY,
        max_tokens=2048
    )

def run_query(query, k=4):
    """
    Run a RAG query against the vector store.
    
    Args:
        query: User's question
        k: Number of documents to retrieve
    
    Returns:
        Answer string
    """
    try:
        # Initialize components
        embeddings = get_embeddings()
        vectordb = init_chroma(embedding_fn=embeddings)
        retriever = vectordb.as_retriever(search_kwargs={"k": k})
        
        # Initialize LLM
        llm = get_llm()
        
        # Create prompt template
        template = """You are a data analyst assistant. Answer the question based on the following data context.

Context:
{context}

Question: {question}

Provide a clear, concise answer with specific data points when available. If you cannot answer based on the context, say so.

Answer:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        # Create RAG chain
        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        # Run the chain
        result = chain.invoke(query)
        return result
        
    except Exception as e:
        return f"Error processing query: {str(e)}"

def generate_insights(df_summary, k=6):
    """
    Generate business insights from data summary.
    
    Args:
        df_summary: Dictionary containing DataFrame summary statistics
        k: Number of insights to generate
    
    Returns:
        Formatted insights string
    """
    try:
        llm = get_llm()
        
        prompt = f"""You are a business intelligence analyst. Based on the data summary below, generate {k} actionable business insights.

Data Summary:
- Shape: {df_summary.get('shape', 'N/A')}
- Columns: {', '.join(df_summary.get('columns', []))}
- Data Types: {df_summary.get('dtypes', {})}
- Missing Values: {df_summary.get('missing_values', {})}
- Statistics: {df_summary.get('summary_stats', {})}

For each insight:
1. State the insight clearly (1-2 sentences)
2. Provide supporting evidence from the data
3. Add a confidence level (High/Medium/Low)
4. Suggest a potential action

Format each insight as:
**Insight X:** [insight statement]
**Evidence:** [supporting data]
**Confidence:** [High/Medium/Low]
**Action:** [recommended action]

Generate {k} insights:"""
        
        response = llm.invoke(prompt)
        return response.content
        
    except Exception as e:
        return f"Error generating insights: {str(e)}"