import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from io import BytesIO
import os

# Page config
st.set_page_config(
    page_title="Data-to-Insights RAG Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_BASE_URL = os.getenv("API_URL", "http://localhost:8000")

# Initialize session state
if 'data_uploaded' not in st.session_state:
    st.session_state.data_uploaded = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'data_summary' not in st.session_state:
    st.session_state.data_summary = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'pending_question' not in st.session_state:
    st.session_state.pending_question = None
if 'cleaning_summary' not in st.session_state:
    st.session_state.cleaning_summary = None

# Function to process query
def process_query(question, k_value):
    """Process a query and return the answer."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/query",
            json={"question": question, "k": k_value},
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            answer = result['answer']
            
            # Fix: Escape dollar signs to prevent LaTeX/math mode rendering
            answer = answer.replace('$', '\\$')
            
            return answer, None
        else:
            return None, f"Error: {response.text}"
    except Exception as e:
        return None, f"Error: {str(e)}"

# Header
st.markdown('<div class="main-header">📊 Data-to-Insights RAG Agent</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Enterprise Analytics Assistant powered by Groq & LangChain</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("📁 Data Upload")
    
    uploaded_file = st.file_uploader(
        "Upload CSV or Excel file",
        type=['csv', 'xlsx', 'xls'],
        help="Upload your data file to start analysis"
    )
    
    if uploaded_file is not None:
        if st.button("🚀 Process Data", use_container_width=True):
            with st.spinner("Processing your data..."):
                try:
                    files = {'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    
                    response = requests.post(
                        f"{API_BASE_URL}/ingest", 
                        files=files,
                        timeout=600
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        st.session_state.data_uploaded = True
                        
                        if uploaded_file.name.endswith('.csv'):
                            st.session_state.df = pd.read_csv(BytesIO(uploaded_file.getvalue()))
                        else:
                            st.session_state.df = pd.read_excel(BytesIO(uploaded_file.getvalue()))
                        
                        st.session_state.data_summary = result['data_summary']
                        
                        if 'cleaning_summary' in result:
                            st.session_state.cleaning_summary = result.get('cleaning_summary', '')
                        
                        st.success(f"✅ Successfully processed {result['ingest_result']['rows_processed']} rows!")
                        
                        if 'cleaning_summary' in result:
                            with st.expander("🧹 Data Cleaning Summary", expanded=True):
                                st.code(result['cleaning_summary'])
                        
                        st.rerun()
                    else:
                        st.error(f"❌ Error: {response.text}")
                        
                except requests.exceptions.Timeout:
                    st.error("❌ Request timed out. Please try again.")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    st.divider()
    
    st.header("⚙️ Settings")
    k_value = st.slider(
        "Number of relevant docs", 
        min_value=1, 
        max_value=5, 
        value=2,
        help="How many data chunks to retrieve (lower = faster, fewer tokens)"
    )
    
    st.divider()
    
    st.header("ℹ️ About")
    st.info("""
    **Features:**
    - 📤 Upload CSV/Excel files
    - 🤖 RAG-powered Q&A
    - 📊 Auto-generated insights
    - 📈 Data visualization
    - 💬 Chat interface
    """)

# Main content
if not st.session_state.data_uploaded:
    st.markdown("""
    <div style="background-color: #d1ecf1; border: 1px solid #bee5eb; color: #0c5460; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0;">
        <h3>👋 Welcome!</h3>
        <p>Upload your CSV or Excel file using the sidebar to get started.</p>
        <p><strong>What you can do:</strong></p>
        <ul>
            <li>Ask questions about your data in natural language</li>
            <li>Get AI-generated business insights</li>
            <li>Visualize your data with interactive charts</li>
            <li>Export analysis results</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("💡 Example Use Cases")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **📈 Sales Analytics**
        - Top performing products
        - Revenue trends
        - Customer segments
        """)
    
    with col2:
        st.markdown("""
        **👥 HR Analytics**
        - Employee turnover
        - Department insights
        - Performance metrics
        """)
    
    with col3:
        st.markdown("""
        **📊 Operations**
        - Process efficiency
        - Resource utilization
        - Cost analysis
        """)

else:
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Overview", "💬 Chat & Query", "💡 Insights", "📈 Visualizations"])
    
    # Tab 1: Data Overview
    with tab1:
        st.header("📊 Data Overview")
        
        if st.session_state.cleaning_summary:
            with st.expander("🧹 Data Cleaning Report", expanded=False):
                st.code(st.session_state.cleaning_summary, language="text")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Rows", st.session_state.data_summary['shape'][0])
        with col2:
            st.metric("Total Columns", st.session_state.data_summary['shape'][1])
        with col3:
            missing_count = sum(st.session_state.data_summary['missing_values'].values())
            st.metric("Missing Values", missing_count)
        with col4:
            numeric_cols = len([k for k, v in st.session_state.data_summary['dtypes'].items() 
                              if 'int' in v or 'float' in v])
            st.metric("Numeric Columns", numeric_cols)
        
        st.divider()
        
        st.subheader("🔍 Data Preview")
        st.dataframe(st.session_state.df.head(20), use_container_width=True)
        
        st.subheader("📋 Column Details")
        col_details = pd.DataFrame({
            'Column': st.session_state.data_summary['columns'],
            'Data Type': [st.session_state.data_summary['dtypes'][col] 
                         for col in st.session_state.data_summary['columns']],
            'Missing Values': [st.session_state.data_summary['missing_values'][col] 
                              for col in st.session_state.data_summary['columns']]
        })
        st.dataframe(col_details, use_container_width=True)
    
    # Tab 2: Chat & Query
    with tab2:
        col_header, col_clear = st.columns([4, 1])
        with col_header:
            st.header("💬 Chat with Your Data")
        with col_clear:
            if len(st.session_state.chat_history) > 0:
                if st.button("🗑️ Clear Chat", use_container_width=True):
                    st.session_state.chat_history = []
                    st.rerun()
        
        # Process pending question
        if st.session_state.pending_question:
            query = st.session_state.pending_question
            st.session_state.pending_question = None
            
            with st.spinner("Thinking..."):
                answer, error = process_query(query, k_value)
                
                if answer:
                    st.session_state.chat_history.append({
                        'question': query,
                        'answer': answer
                    })
                    st.rerun()
                else:
                    st.error(error)
        
        # Chat history
        for chat in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(chat['question'])
            with st.chat_message("assistant"):
                st.write(chat['answer'])
        
        # Suggested questions
        if len(st.session_state.chat_history) == 0:
            st.subheader("💡 Suggested Questions")
            st.caption("Click any question below to get started:")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📊 What are the main trends in the data?", use_container_width=True, key="q1"):
                    st.session_state.pending_question = "What are the main trends in the data?"
                    st.rerun()
                if st.button("📈 Show me summary statistics", use_container_width=True, key="q2"):
                    st.session_state.pending_question = "Show me summary statistics"
                    st.rerun()
            
            with col2:
                if st.button("🔍 What insights can you find?", use_container_width=True, key="q3"):
                    st.session_state.pending_question = "What insights can you find in this data?"
                    st.rerun()
                if st.button("❓ What columns have missing data?", use_container_width=True, key="q4"):
                    st.session_state.pending_question = "What columns have missing data?"
                    st.rerun()
            
            st.divider()
        
        # Chat input
        query = st.chat_input("Ask a question about your data...")
        
        if query:
            with st.chat_message("user"):
                st.write(query)
            
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer, error = process_query(query, k_value)
                    
                    if answer:
                        st.write(answer)
                        st.session_state.chat_history.append({
                            'question': query,
                            'answer': answer
                        })
                    else:
                        st.error(error)
    
    # Tab 3: Insights - FIXED!
    with tab3:
        st.header("💡 AI-Generated Business Insights")
        
        if st.button("🎯 Generate Insights", use_container_width=True):
            with st.spinner("Analyzing your data and generating insights..."):
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/insights",
                        json={"df_summary": st.session_state.data_summary, "k": 6},
                        timeout=120
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        insights_text = result['insights']
                        
                        # Escape dollar signs in insights too
                        insights_text = insights_text.replace('$', '\\$')
                        
                        st.markdown("---")
                        
                        # Split by "**Insight" to separate each insight
                        insight_sections = insights_text.split('**Insight')
                        
                        for idx, section in enumerate(insight_sections[1:], 1):
                            section = '**Insight' + section
                            section = section.strip()
                            
                            # Create expander
                            with st.expander(f"💡 Insight {idx}", expanded=(idx <= 2)):
                                # Render as markdown (not HTML) - this fixes the formatting!
                                st.markdown(section)
                        
                        st.markdown("---")
                        
                        st.download_button(
                            label="📥 Download Insights Report",
                            data=insights_text,
                            file_name="business_insights.txt",
                            mime="text/plain",
                            use_container_width=True
                        )
                    else:
                        st.error(f"Error: {response.text}")
                except requests.exceptions.Timeout:
                    st.error("⏱️ Request timed out. Please try again.")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    # Tab 4: Visualizations
    with tab4:
        st.header("📈 Data Visualizations")
        
        numeric_cols = [col for col in st.session_state.df.columns 
                       if pd.api.types.is_numeric_dtype(st.session_state.df[col])]
        
        if len(numeric_cols) > 0:
            st.subheader("📊 Distribution Analysis")
            selected_col = st.selectbox("Select column", numeric_cols)
            
            fig = px.histogram(st.session_state.df, x=selected_col, 
                             title=f"Distribution of {selected_col}",
                             marginal="box")
            st.plotly_chart(fig, use_container_width=True)
            
            if len(numeric_cols) > 1:
                st.subheader("🔥 Correlation Heatmap")
                corr_matrix = st.session_state.df[numeric_cols].corr()
                
                fig = px.imshow(corr_matrix, 
                              text_auto=True,
                              title="Correlation Matrix",
                              color_continuous_scale='RdBu_r')
                st.plotly_chart(fig, use_container_width=True)
            
            date_cols = [col for col in st.session_state.df.columns 
                        if pd.api.types.is_datetime64_any_dtype(st.session_state.df[col])]
            
            if len(date_cols) > 0 and len(numeric_cols) > 0:
                st.subheader("📅 Time Series Analysis")
                date_col = st.selectbox("Select date column", date_cols)
                value_col = st.selectbox("Select value column", numeric_cols)
                
                fig = px.line(st.session_state.df, x=date_col, y=value_col,
                            title=f"{value_col} over time")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No numeric columns found for visualization.")

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    Built with ❤️ using Streamlit, LangChain & Groq | Data-to-Insights RAG Agent v1.0
</div>
""", unsafe_allow_html=True)