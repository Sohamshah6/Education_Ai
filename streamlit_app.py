import streamlit as st
import os
import tempfile
from typing import Optional, List, Dict, Any
import time
from datetime import datetime
import json
import io
#import streamlit as st

# Import your backend classes
from rag_chatbot import RAGChatbot, QuizQuestion, QuizResult

# Page configuration
st.set_page_config(
    page_title="Education Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .quiz-question {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .quiz-option {
        background-color: white;
        padding: 0.8rem;
        margin: 0.5rem 0;
        border-radius: 5px;
        border: 1px solid #e0e0e0;
    }
    
    .correct-answer {
        background-color: #d4edda;
        border-color: #c3e6cb;
        color: #155724;
    }
    
    .incorrect-answer {
        background-color: #f8d7da;
        border-color: #f5c6cb;
        color: #721c24;
    }
    
    .quiz-result {
        background-color: #e8f5e8;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
    }
    
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    
    .bot-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    
    .sidebar-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    """Initialize session state variables."""
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'quiz_active' not in st.session_state:
        st.session_state.quiz_active = False
    if 'quiz_questions' not in st.session_state:
        st.session_state.quiz_questions = []
    if 'current_question_index' not in st.session_state:
        st.session_state.current_question_index = 0
    if 'quiz_answers' not in st.session_state:
        st.session_state.quiz_answers = []
    if 'quiz_completed' not in st.session_state:
        st.session_state.quiz_completed = False
    if 'quiz_results' not in st.session_state:
        st.session_state.quiz_results = None
    if 'show_quiz_review' not in st.session_state:
        st.session_state.show_quiz_review = False
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = []

def initialize_chatbot():
    """Initialize the chatbot with API key."""
    try:
        # Check if API key is in environment or get from user
        api_key = os.getenv("GOOGLE_API_KEY")
        
        if not api_key:
            st.error("⚠️ GOOGLE_API_KEY not found in environment variables!")
            st.info("Please set your Google API key in the sidebar.")
            return False
        
        if st.session_state.chatbot is None:
            with st.spinner("🤖 Initializing chatbot..."):
                st.session_state.chatbot = RAGChatbot(api_key=api_key)
            st.success("✅ Chatbot initialized successfully!")
        
        return True
    except Exception as e:
        st.error(f"❌ Error initializing chatbot: {str(e)}")
        return False



def process_uploaded_file(uploaded_file):
    """Process uploaded PDF file using in-memory approach (Streamlit Cloud safe)."""
    try:
        if uploaded_file is not None:
            # Read uploaded PDF file in memory as bytes
            pdf_bytes = uploaded_file.read()
            pdf_stream = io.BytesIO(pdf_bytes)

            with st.spinner(f"📄 Processing {uploaded_file.name}..."):
                # Pass the in-memory PDF stream to your chatbot logic
                success = st.session_state.chatbot.process_pdf(pdf_stream)

            if success:
                st.success(f"✅ Successfully processed {uploaded_file.name}")
                st.session_state.processed_files = st.session_state.chatbot.get_processed_files()
                return True
            else:
                st.error(f"❌ Failed to process {uploaded_file.name}")
                return False

    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.exception(e)  # Optional: shows traceback for debugging
        return False


def display_chat_interface():
    """Display the chat interface."""
    st.header("💬 Chat with Your Document")
    
    # Check if documents are loaded
    if not st.session_state.processed_files:
        st.warning("⚠️ Please upload and process a PDF document first.")
        return
    
    # Display processed files
    st.info(f"📚 Loaded documents: {', '.join(st.session_state.processed_files)}")
    
    # Chat input
    user_input = st.text_input("💬 Ask a question about your document:", key="chat_input")
    
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🚀 Send", key="send_button"):
            if user_input.strip():
                handle_chat_input(user_input)
    
    with col2:
        if st.button("🗑️ Clear Chat", key="clear_chat"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Display chat history
    if st.session_state.chat_history:
        st.subheader("💭 Chat History")
        for i, (question, answer) in enumerate(st.session_state.chat_history):
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>🧑 You:</strong> {question}
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="chat-message bot-message">
                <strong>🤖 Bot:</strong> {answer}
            </div>
            """, unsafe_allow_html=True)

def handle_chat_input(user_input):
    """Handle user chat input."""
    try:
        with st.spinner("🤔 Thinking..."):
            response = st.session_state.chatbot.chat(user_input)
        
        # Add to chat history
        st.session_state.chat_history.append((user_input, response))
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Error processing your question: {str(e)}")

def display_quiz_interface():
    """Display the quiz interface."""
    st.header("🎯 Quiz Section")
    
    # Check if documents are loaded
    if not st.session_state.processed_files:
        st.warning("⚠️ Please upload and process a PDF document first.")
        return
    
    # Quiz configuration section
    if not st.session_state.quiz_active:
        st.subheader("🎲 Create New Quiz")
        
        col1, col2 = st.columns(2)
        
        with col1:
            topic = st.text_input("🎯 Topic (optional):", help="Leave empty for general quiz")
            num_questions = st.number_input("📊 Number of questions:", min_value=1, max_value=20, value=5)
        
        with col2:
            difficulty = st.selectbox("🎚️ Difficulty:", ["easy", "medium", "hard"], index=1)
            question_type = st.selectbox("❓ Question Type:", ["multiple_choice", "true_false"], index=0)
        
        if st.button("🎯 Generate Quiz", key="generate_quiz"):
            create_quiz(topic, num_questions, difficulty, question_type)
    
    # Active quiz section
    if st.session_state.quiz_active and not st.session_state.quiz_completed:
        display_active_quiz()
    
    # Quiz results section
    if st.session_state.quiz_completed:
        display_quiz_results()
    
    # Quiz history section
    display_quiz_history()

def create_quiz(topic, num_questions, difficulty, question_type):
    """Create a new quiz."""
    try:
        with st.spinner("🎲 Generating quiz questions..."):
            success = st.session_state.chatbot.create_quiz(
                topic=topic,
                num_questions=num_questions,
                difficulty=difficulty,
                question_type=question_type
            )
        
        if success:
            st.success("✅ Quiz created successfully!")
            st.session_state.quiz_active = True
            st.session_state.quiz_completed = False
            st.session_state.quiz_answers = []
            st.session_state.current_question_index = 0
            st.session_state.quiz_results = None
            st.session_state.show_quiz_review = False
            st.rerun()
        else:
            st.error("❌ Failed to create quiz. Please try again.")
    
    except Exception as e:
        st.error(f"❌ Error creating quiz: {str(e)}")

def display_active_quiz():
    """Display the active quiz."""
    question_data = st.session_state.chatbot.get_current_quiz_question()
    
    if not question_data:
        st.session_state.quiz_active = False
        st.session_state.quiz_completed = True
        st.rerun()
        return
    
    progress = question_data['progress']
    
    # Progress bar
    st.progress(progress['current_question'] / progress['total_questions'])
    
    # Question display
    st.markdown(f"""
    <div class="quiz-question">
        <h3>Question {progress['current_question']} of {progress['total_questions']}</h3>
        <p><strong>Difficulty:</strong> {question_data['difficulty'].title()}</p>
        <h4>{question_data['question']}</h4>
    </div>
    """, unsafe_allow_html=True)
    
    # Options
    st.subheader("Choose your answer:")
    
    # Create radio buttons for options
    options = question_data['options']
    
    if question_data['question_type'] == 'multiple_choice':
        # For multiple choice, show A, B, C, D format
        selected_option = st.radio(
            "Select an option:",
            options,
            key=f"quiz_option_{progress['current_question']}"
        )
        
        # Extract letter from selected option (A, B, C, D)
        if selected_option:
            answer = selected_option[0]  # Get the letter (A, B, C, D)
    else:
        # For true/false questions
        selected_option = st.radio(
            "Select an option:",
            options,
            key=f"quiz_option_{progress['current_question']}"
        )
        answer = selected_option
    
    # Submit button
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        if st.button("✅ Submit Answer", key=f"submit_{progress['current_question']}"):
            if 'selected_option' in locals():
                submit_quiz_answer(answer)
            else:
                st.warning("Please select an answer before submitting.")
    
    # Quit quiz button
    with col3:
        if st.button("❌ Quit Quiz", key="quit_quiz"):
            st.session_state.quiz_active = False
            st.session_state.quiz_completed = False
            st.rerun()

def submit_quiz_answer(answer):
    """Submit quiz answer."""
    try:
        result = st.session_state.chatbot.submit_quiz_answer(answer)
        
        # Show immediate feedback
        if result['is_correct']:
            st.success("✅ Correct!")
        else:
            st.error("❌ Incorrect!")
            if 'correct_answer' in result:
                st.info(f"Correct answer: {result['correct_answer']}")
            if 'explanation' in result:
                st.info(f"Explanation: {result['explanation']}")
        
        # Store answer for review
        st.session_state.quiz_answers.append(result)
        
        # Check if quiz is completed
        if result['quiz_completed']:
            st.session_state.quiz_active = False
            st.session_state.quiz_completed = True
            st.session_state.quiz_results = result.get('final_results')
        
        # Wait a bit before moving to next question
        time.sleep(2)
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Error submitting answer: {str(e)}")

def display_quiz_results():
    """Display quiz results."""
    if not st.session_state.quiz_results:
        return
    
    results = st.session_state.quiz_results
    
    st.markdown(f"""
    <div class="quiz-result">
        <h2>🎉 Quiz Completed!</h2>
        <h3>📊 Your Score: {results['correct_answers']}/{results['total_questions']}</h3>
        <h3>📈 Percentage: {results['score_percentage']:.1f}%</h3>
        <h3>⏱️ Time Taken: {results['time_taken']}</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Performance message
    if results['score_percentage'] >= 80:
        st.balloons()
        st.success("🌟 Excellent performance! You have a great understanding of the material.")
    elif results['score_percentage'] >= 60:
        st.success("👍 Good job! You have a solid grasp of the concepts.")
    else:
        st.warning("💪 Keep studying! Review the material and try again.")
    
    # Buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔍 View Review", key="view_review"):
            st.session_state.show_quiz_review = True
            st.rerun()
    
    with col2:
        if st.button("🔄 New Quiz", key="new_quiz"):
            st.session_state.quiz_active = False
            st.session_state.quiz_completed = False
            st.session_state.quiz_results = None
            st.session_state.show_quiz_review = False
            st.rerun()
    
    with col3:
        if st.button("📊 Quiz History", key="view_history"):
            st.session_state.show_quiz_review = False
    
    # Quiz review section
    if st.session_state.show_quiz_review:
        display_quiz_review()

def display_quiz_review():
    """Display detailed quiz review."""
    st.subheader("📋 Quiz Review")
    
    try:
        review_data = st.session_state.chatbot.get_quiz_review()
        
        if not review_data:
            st.warning("No review data available.")
            return
        
        for i, item in enumerate(review_data, 1):
            status_icon = "✅" if item['is_correct'] else "❌"
            status_class = "correct-answer" if item['is_correct'] else "incorrect-answer"
            
            st.markdown(f"""
            <div class="quiz-option {status_class}">
                <h4>{i}. {item['question']}</h4>
                <p><strong>Your Answer:</strong> {item['user_answer']}</p>
                <p><strong>Correct Answer:</strong> {item['correct_answer']} {status_icon}</p>
                <p><strong>Explanation:</strong> {item['explanation']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"❌ Error displaying review: {str(e)}")

def display_quiz_history():
    """Display quiz history."""
    st.subheader("📈 Quiz History")
    
    try:
        history = st.session_state.chatbot.get_quiz_history() if st.session_state.chatbot else []
        
        if not history:
            st.info("📭 No quiz history available.")
            return
        
        # Create a table for history
        history_data = []
        for i, result in enumerate(history, 1):
            history_data.append({
                "Quiz #": i,
                "Score": f"{result.correct_answers}/{result.total_questions}",
                "Percentage": f"{result.score_percentage:.1f}%",
                "Time": result.time_taken
            })
        
        st.table(history_data)
    
    except Exception as e:
        st.error(f"❌ Error displaying history: {str(e)}")

def display_sidebar():
    """Display the sidebar with settings and information."""
    st.sidebar.markdown("## ⚙️ Settings")
    
    # API Key section
    st.sidebar.markdown("### 🔑 API Configuration")
    api_key = st.sidebar.text_input("Google API Key:", type="password", help="Enter your Google API key")
    
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
        st.sidebar.success("✅ API key set!")
    
    # File upload section
    st.sidebar.markdown("### 📁 Document Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Upload PDF document",
        type=['pdf'],
        help="Upload a PDF file to chat with or create quizzes from"
    )
    
    if uploaded_file is not None:
        if st.sidebar.button("📤 Process Document"):
            if st.session_state.chatbot is None:
                if not initialize_chatbot():
                    return
            
            if process_uploaded_file(uploaded_file):
                st.sidebar.success("✅ Document processed successfully!")
            else:
                st.sidebar.error("❌ Failed to process document")
    
    # Processed files section
    if st.session_state.processed_files:
        st.sidebar.markdown("### 📚 Processed Files")
        for file in st.session_state.processed_files:
            st.sidebar.info(f"📄 {file}")
    
    # Database management
    st.sidebar.markdown("### 🗄️ Database Management")
    if st.sidebar.button("🗑️ Clear Database"):
        if st.session_state.chatbot:
            if st.session_state.chatbot.clear_database():
                st.sidebar.success("✅ Database cleared!")
                st.session_state.processed_files = []
                st.session_state.chat_history = []
                st.session_state.quiz_active = False
                st.session_state.quiz_completed = False
            else:
                st.sidebar.error("❌ Failed to clear database")
    
    # About section
    st.sidebar.markdown("### ℹ️ About")
    st.sidebar.info("""
    This is a RAG (Retrieval-Augmented Generation) chatbot with quiz functionality.
    
    Features:
    - 💬 Chat with your PDF documents
    - 🎯 Generate quizzes from document content
    - 📊 Track quiz performance
    - 📈 View quiz history
    """)

def main():
    """Main application function."""
    # Initialize session state
    init_session_state()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Educational Agent Your Own Guide</h1>
        <p>Upload your PDF documents, chat with them, and test your knowledge with AI-generated quizzes!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize chatbot if not already done
    if st.session_state.chatbot is None:
        initialize_chatbot()
    
    # Sidebar
    display_sidebar()
    
    # Main content tabs
    tab1, tab2 = st.tabs(["💬 Chat", "🎯 Quiz"])
    
    with tab1:
        display_chat_interface()
    
    with tab2:
        display_quiz_interface()

if __name__ == "__main__":
    main()
