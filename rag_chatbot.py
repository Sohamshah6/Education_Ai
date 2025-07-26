# enhanced_rag_chatbot.py
import os
import shutil
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
import logging
from typing import List, Optional, Tuple, Dict, Any
import tempfile
import uuid
import json
import re
import random
from dataclasses import dataclass
from datetime import datetime
import io
import fitz

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class QuizQuestion:
    """Data class for quiz questions."""
    question: str
    options: List[str]
    correct_answer: str
    explanation: str
    difficulty: str = "medium"  # easy, medium, hard
    question_type: str = "multiple_choice"  # multiple_choice, true_false, fill_blank

@dataclass
class QuizResult:
    """Data class for quiz results."""
    total_questions: int
    correct_answers: int
    incorrect_answers: int
    score_percentage: float
    time_taken: str
    questions_results: List[Dict[str, Any]]

class QuizGenerator:
    """Handles quiz generation from document content."""
    
    def __init__(self, llm, vectorstore):
        self.llm = llm
        self.vectorstore = vectorstore
    
    def generate_quiz_questions(self, topic: str = "", num_questions: int = 5, 
                            difficulty: str = "medium", question_type: str = "multiple_choice") -> List[QuizQuestion]:
        """Generate quiz questions based on document content."""
        try:
            logger.info(f"🎯 Generating quiz: topic='{topic}', num_questions={num_questions}, difficulty={difficulty}")
            
            # Get relevant documents
            relevant_docs = []
            
            # Check if vectorstore has any documents
            if not self.vectorstore:
                logger.error("❌ Vectorstore is None")
                return []
            
            # Try to get documents based on topic or general content
            if topic and topic.strip():
                logger.info(f"📖 Searching for topic: '{topic}'")
                try:
                    relevant_docs = self.vectorstore.similarity_search(topic, k=8)
                except Exception as e:
                    logger.warning(f"⚠️ Error searching for topic '{topic}': {e}")
                    relevant_docs = []
                
                if not relevant_docs:
                    logger.warning(f"⚠️ No relevant documents found for topic '{topic}', trying general search")
            
            # If no topic-specific results, get general documents
            if not relevant_docs:
                try:
                    # Use a more general approach to get documents
                    logger.info("📚 Performing general document search")
                    relevant_docs = self.vectorstore.similarity_search("content", k=10)
                    
                    # If still no results, try getting all documents
                    if not relevant_docs:
                        logger.info("📄 Trying to get all available documents")
                        # Get documents by trying different common terms
                        search_terms = ["the", "and", "a", "an", "is", "are", "was", "were", "text", "document"]
                        for term in search_terms:
                            try:
                                relevant_docs = self.vectorstore.similarity_search(term, k=20)
                                if relevant_docs:
                                    logger.info(f"✅ Found documents using search term: '{term}'")
                                    break
                            except Exception as e:
                                logger.warning(f"⚠️ Error searching with term '{term}': {e}")
                                continue
                        
                        # Last resort: try to get documents using vectorstore's get method
                        if not relevant_docs:
                            try:
                                # Try to access the collection directly
                                collection = self.vectorstore._collection
                                if hasattr(collection, 'get'):
                                    result = collection.get(limit=20)
                                    if result and result.get('documents'):
                                        # Create mock documents from the raw data
                                        from langchain.schema import Document
                                        relevant_docs = [
                                            Document(page_content=doc, metadata={})
                                            for doc in result['documents']
                                            if doc and doc.strip()
                                        ]
                                        logger.info(f"✅ Retrieved {len(relevant_docs)} documents from collection")
                            except Exception as e:
                                logger.warning(f"⚠️ Error accessing collection directly: {e}")
                    
                except Exception as e:
                    logger.error(f"❌ Error in general document search: {e}")
                    return []

            # Filter documents that have valid, non-empty page_content
            filtered_docs = []
            for doc in relevant_docs:
                if hasattr(doc, 'page_content') and doc.page_content and doc.page_content.strip():
                    filtered_docs.append(doc)
            
            logger.info(f"📊 Found {len(filtered_docs)} valid documents after filtering")
            
            if not filtered_docs:
                logger.error("❌ No valid documents found after filtering. Cannot generate quiz.")
                return []

            # Combine into one string and limit size
            content_parts = []
            total_length = 0
            max_content_length = 15000  # Limit to prevent token issues
            
            for doc in filtered_docs:
                content = doc.page_content.strip()
                if total_length + len(content) > max_content_length:
                    # Add partial content to stay under limit
                    remaining = max_content_length - total_length
                    if remaining > 100:  # Only add if there's meaningful space left
                        content_parts.append(content[:remaining])
                    break
                content_parts.append(content)
                total_length += len(content)
            
            combined_content = "\n\n".join(content_parts)
            
            if not combined_content.strip():
                logger.error("❌ Combined content is empty after processing. Cannot generate quiz.")
                return []

            logger.info(f"📄 Quiz content length: {len(combined_content)} characters")
            logger.info(f"📄 Content preview: {combined_content[:200]}...")

            # Build and send prompt
            quiz_prompt = self._build_quiz_prompt(combined_content, num_questions, difficulty, question_type)
            
            logger.info("🤖 Sending prompt to LLM...")
            response = self.llm.invoke(quiz_prompt)
            
            if not response or not response.content:
                logger.error("❌ Empty response from LLM")
                return []

            logger.info("📝 Parsing LLM response...")
            # Parse the response
            questions = self._parse_quiz_response(response.content)

            logger.info(f"✅ Generated {len(questions)} quiz questions.")
            return questions

        except Exception as e:
            logger.error(f"❌ Error generating quiz questions: {e}")
            import traceback
            logger.error(f"❌ Traceback: {traceback.format_exc()}")
            return []

    def _build_quiz_prompt(self, content: str, num_questions: int, difficulty: str, question_type: str) -> str:
        """Build prompt for quiz generation."""
        prompt = f"""
    Based on the following document content, create {num_questions} {difficulty} {question_type.replace('_', ' ')} questions.

Document Content:
{content}

Requirements:
- Generate exactly {num_questions} questions
- Difficulty level: {difficulty}
- Question type: {question_type.replace('_', ' ')}
- Each question should test understanding of the content
- For multiple choice questions, provide exactly 4 options (A, B, C, D)
- For true/false questions, provide exactly 2 options (True, False)
- Include explanations for correct answers
- Focus on key concepts, facts, and important details

IMPORTANT: Return ONLY a valid JSON array. Do not include any markdown code blocks, citations, or extra text.

Format your response as a JSON array with the following structure:
[
  {{
    "question": "Question text here?",
    "options": ["A) Option 1", "B) Option 2", "C) Option 3", "D) Option 4"],
    "correct_answer": "A",
    "explanation": "Explanation of why this answer is correct",
    "difficulty": "{difficulty}",
    "question_type": "{question_type}"
  }}
]

For true/false questions, use: "options": ["True", "False"]

Rules:
1. Return ONLY valid JSON - no markdown blocks, no extra text
2. Do not include citations or references within the JSON structure
3. Keep explanations concise and factual
4. Ensure all JSON syntax is correct (proper commas, brackets, quotes)
5. Do not add any text before or after the JSON array
"""
        return prompt
    
    def _parse_quiz_response(self, response: str) -> List[QuizQuestion]:
        """Parse LLM response to extract quiz questions."""
        try:
            logger.info("🔍 Parsing quiz response...")
        
            # Clean the response - remove markdown code blocks if present
            cleaned_response = response.strip()
        
            # Remove markdown code blocks
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]  # Remove ```json
            if cleaned_response.startswith("```"):
                cleaned_response = cleaned_response[3:]   # Remove ```
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]  # Remove trailing ```
        
            cleaned_response = cleaned_response.strip()
        
            # Try to extract JSON from response
            start_idx = cleaned_response.find('[')
            end_idx = cleaned_response.rfind(']') + 1
        
            if start_idx == -1 or end_idx == 0:
                logger.error("❌ No JSON array found in response")
                logger.error(f"Response preview: {cleaned_response[:500]}...")
                return []
        
            json_str = cleaned_response[start_idx:end_idx]
            logger.info(f"📝 Extracted JSON: {json_str[:200]}...")
        
            # Clean up any non-printable characters
            json_str = re.sub(r'[^\x20-\x7E\n\r\t]+', '', json_str)
        
            # Additional cleaning for malformed JSON
            # Fix common issues like extra text in arrays
            lines = json_str.split('\n')
            cleaned_lines = []
            in_options_array = False
        
            for line in lines:
                stripped_line = line.strip()
            
                # Check if we're entering an options array
                if '"options":' in stripped_line:
                    in_options_array = True
                    cleaned_lines.append(line)
                    continue
            
                # Check if we're exiting an options array
                if in_options_array and stripped_line.startswith(']'):
                    in_options_array = False
                    cleaned_lines.append(line)
                    continue
            
                # If we're in options array, only keep lines that look like valid options
                if in_options_array:
                    # Skip lines that don't look like options (citations, extra text, etc.)
                    if (stripped_line.startswith('"') and 
                        (stripped_line.startswith('"A)') or 
                        stripped_line.startswith('"B)') or 
                        stripped_line.startswith('"C)') or 
                        stripped_line.startswith('"D)') or 
                        stripped_line.startswith('"True') or 
                        stripped_line.startswith('"False'))):
                        cleaned_lines.append(line)
                    elif stripped_line in ['"True",', '"False",', '"True"', '"False"']:
                        cleaned_lines.append(line)
                    # Skip other lines (like citations or malformed content)
                    continue
                else:
                    cleaned_lines.append(line)
        
            json_str = '\n'.join(cleaned_lines)
        
            logger.info(f"📝 Cleaned JSON: {json_str[:300]}...")
        
            try:
                questions_data = json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.error(f"❌ JSON still malformed after cleaning: {e}")
                logger.error(f"Problematic JSON: {json_str}")
            
                # Try a more aggressive approach - rebuild the JSON
                questions_data = self._rebuild_json_from_text(cleaned_response)
                if not questions_data:
                    return []
        
            if not isinstance(questions_data, list):
                logger.error("❌ Response is not a list")
                return []
        
            questions = []
            for i, q_data in enumerate(questions_data):
                try:
                    question = QuizQuestion(
                        question=q_data.get('question', ''),
                        options=q_data.get('options', []),
                        correct_answer=q_data.get('correct_answer', ''),
                        explanation=q_data.get('explanation', ''),
                        difficulty=q_data.get('difficulty', 'medium'),
                        question_type=q_data.get('question_type', 'multiple_choice')
                    )
                
                    # Validate question
                    if not question.question.strip():
                        logger.warning(f"⚠️ Question {i+1} has empty question text")
                        continue
                
                    if not question.options:
                        logger.warning(f"⚠️ Question {i+1} has no options")
                        continue
                
                    questions.append(question)
                    logger.info(f"✅ Parsed question {i+1}: {question.question[:50]}...")
                
                except Exception as e:
                    logger.error(f"❌ Error parsing question {i+1}: {e}")
                    continue
        
            logger.info(f"✅ Successfully parsed {len(questions)} questions")
            return questions
        
        except Exception as e:
            logger.error(f"❌ Error parsing quiz response: {e}")
            logger.error(f"❌ Full response: {response}")
            return []

    def _rebuild_json_from_text(self, text: str) -> List[Dict]:
        """Attempt to rebuild JSON structure from malformed text."""
        try:
            logger.info("🔧 Attempting to rebuild JSON from text...")
            
            # Use regex to find question blocks
            question_pattern = r'"question":\s*"([^"]+)"'
            #options_pattern = r'"options":\s*\[(.*?)\]'
            answer_pattern = r'"correct_answer":\s*"([^"]+)"'
            explanation_pattern = r'"explanation":\s*"([^"]+)"'
            
            questions = []
            
            # Split by question blocks
            blocks = re.split(r'\{\s*"question":', text)
            
            for i, block in enumerate(blocks[1:], 1):  # Skip first empty block
                try:
                    # Extract question
                    question_match = re.search(r'^[^"]*"([^"]+)"', block)
                    if not question_match:
                        continue
                    question_text = question_match.group(1)
                    
                    # Extract options
                    options_match = re.search(r'"options":\s*\[(.*?)\]', block, re.DOTALL)
                    if not options_match:
                        continue
                    
                    options_text = options_match.group(1)
                    # Clean and extract individual options
                    options = []
                    for option in re.findall(r'"([^"]+)"', options_text):
                        if option.startswith(('A)', 'B)', 'C)', 'D)')) or option in ['True', 'False']:
                            options.append(option)
                    
                    # Extract correct answer
                    answer_match = re.search(answer_pattern, block)
                    if not answer_match:
                        continue
                    correct_answer = answer_match.group(1)
                    
                    # Extract explanation
                    explanation_match = re.search(explanation_pattern, block)
                    explanation = explanation_match.group(1) if explanation_match else "No explanation provided"
                    
                    questions.append({
                        'question': question_text,
                        'options': options,
                        'correct_answer': correct_answer,
                        'explanation': explanation,
                        'difficulty': 'easy',
                        'question_type': 'multiple_choice'
                    })
                    
                    logger.info(f"✅ Rebuilt question {i}: {question_text[:50]}...")
                    
                except Exception as e:
                    logger.error(f"❌ Error rebuilding question {i}: {e}")
                    continue
            
            logger.info(f"✅ Successfully rebuilt {len(questions)} questions")
            return questions
            
        except Exception as e:
            logger.error(f"❌ Error rebuilding JSON: {e}")
            return []

class QuizManager:
    """Manages quiz sessions and scoring."""
    
    def __init__(self):
        self.current_quiz = []
        self.quiz_results = []
        self.current_question_index = 0
        self.user_answers = []
        self.start_time = None
    
    def start_quiz(self, questions: List[QuizQuestion]):
        """Start a new quiz session."""
        self.current_quiz = questions
        self.current_question_index = 0
        self.user_answers = []
        self.start_time = datetime.now()
        logger.info(f"🎯 Started quiz with {len(questions)} questions")
    
    def get_current_question(self) -> Optional[QuizQuestion]:
        """Get the current question."""
        if self.current_question_index < len(self.current_quiz):
            return self.current_quiz[self.current_question_index]
        return None
    
    # Update the submit_answer method in QuizManager class
    def submit_answer(self, answer: str) -> bool:
        """Submit answer for current question."""
        if self.current_question_index < len(self.current_quiz):
            current_question = self.current_quiz[self.current_question_index]
            
            # Clean the user's answer
            user_answer = answer.strip().upper()
            correct_answer = current_question.correct_answer.strip().upper()
            
            # Handle different answer formats
            if current_question.question_type == "multiple_choice":
                # Extract just the letter (A, B, C, D) from both answers
                if user_answer in ['A', 'B', 'C', 'D']:
                    # User provided just the letter
                    is_correct = user_answer == correct_answer
                else:
                    # User might have provided full option text
                    # Find which option matches user's answer
                    is_correct = False
                    for i, option in enumerate(current_question.options):
                        option_text = option.strip().upper()
                        if user_answer in option_text or option_text in user_answer:
                            user_letter = chr(65 + i)  # Convert to A, B, C, D
                            is_correct = user_letter == correct_answer
                            break
            else:
                # For true/false questions
                is_correct = user_answer == correct_answer
            
            # Store the answer
            self.user_answers.append({
                'question': current_question.question,
                'user_answer': answer,
                'correct_answer': current_question.correct_answer,
                'is_correct': is_correct,
                'explanation': current_question.explanation
            })
            
            # Move to next question
            self.current_question_index += 1
            
            return is_correct
        
        return False

    # Also update the get_quiz_progress method to fix the logging:
    def get_quiz_progress(self) -> Dict[str, Any]:
        """Get current quiz progress."""
        completed = self.current_question_index >= len(self.current_quiz)
        
        progress = {
            'current_question': self.current_question_index + 1,
            'total_questions': len(self.current_quiz),
            'completed': completed
        }
        
        # Only log if not completed to avoid confusion
        if not completed:
            logger.info(f"Quiz progress: {self.current_question_index}/{len(self.current_quiz)}")
        
        return progress
    
    def calculate_results(self) -> QuizResult:
        """Calculate and return quiz results."""
        if not self.user_answers:
            return QuizResult(0, 0, 0, 0.0, "0s", [])
        
        correct_count = sum(1 for answer in self.user_answers if answer['is_correct'])
        total_questions = len(self.user_answers)
        score_percentage = (correct_count / total_questions) * 100
        
        end_time = datetime.now()
        time_taken = str(end_time - self.start_time).split('.')[0]  # Remove microseconds
        
        result = QuizResult(
            total_questions=total_questions,
            correct_answers=correct_count,
            incorrect_answers=total_questions - correct_count,
            score_percentage=score_percentage,
            time_taken=time_taken,
            questions_results=self.user_answers.copy()
        )
        
        self.quiz_results.append(result)
        return result
    
    def get_quiz_history(self) -> List[QuizResult]:
        """Get history of all quiz results."""
        return self.quiz_results.copy()

class RAGChatbot:
    def __init__(self, api_key: Optional[str] = None):
        # Load environment variables
        load_dotenv()
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables or passed as parameter")
        
        self.embedding_model = None
        self.vectorstore = None
        self.processed_files = []  # Track processed files
        self.persist_path = f"./chroma_db_{uuid.uuid4().hex[:8]}"  # Unique DB path
        
        # Initialize quiz components
        self.quiz_generator = None
        self.quiz_manager = QuizManager()
        
        # Initialize LLM with error handling
        try:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash-thinking-exp-01-21",
                temperature=0.2,
                top_p=0.95,
                google_api_key=self.api_key
            )
            logger.info("✅ Gemini LLM initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Gemini LLM: {e}")
            # Fallback to a different model
            try:
                self.llm = ChatGoogleGenerativeAI(
                    model="gemini-pro",
                    temperature=0.2,
                    google_api_key=self.api_key
                )
                logger.info("✅ Fallback Gemini Pro LLM initialized")
            except Exception as e2:
                logger.error(f"❌ Failed to initialize fallback LLM: {e2}")
                raise ValueError("Could not initialize any Gemini model")
    
    def parse_pdf(self, pdf_path: str) -> Optional[str]:
        """Extract text from PDF file using PyMuPDF."""
        try:
            if not os.path.exists(pdf_path):
                logger.error(f"PDF file not found: {pdf_path}")
                return None
                        
            # Open PDF with PyMuPDF
            pdf_document = fitz.open(pdf_path)
            text = ""
            
            # Extract text from each page
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                page_text = page.get_text()
                if page_text:
                    text += page_text + "\n"
            
            pdf_document.close()
                        
            if not text.strip():
                logger.warning("No text extracted from PDF")
                return None
                        
            logger.info(f"✅ Parsed {len(text)} characters from PDF.")
            return text
            
        except Exception as e:
            logger.error(f"❌ Error parsing PDF: {e}")
            return None
    
    def create_chunks(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks."""
        try:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            chunks = splitter.split_text(text)
            
            # Filter out empty chunks
            filtered_chunks = [chunk for chunk in chunks if chunk.strip()]
            
            logger.info(f"✅ Created {len(filtered_chunks)} text chunks (filtered from {len(chunks)}).")
            return filtered_chunks
        except Exception as e:
            logger.error(f"❌ Error creating chunks: {e}")
            return []
    
    def setup_vectorstore(self, text_chunks: List[str]) -> bool:
        """Create embeddings and store in ChromaDB."""
        try:
            # Filter out empty or whitespace-only chunks
            text_chunks = [chunk for chunk in text_chunks if chunk.strip()]
            if not text_chunks:
                logger.error("❌ No valid content to embed. All chunks were empty.")
                return False

            logger.info(f"📊 Setting up vectorstore with {len(text_chunks)} chunks")

            # Initialize embedding model if not already done
            if not self.embedding_model:
                self.embedding_model = GoogleGenerativeAIEmbeddings(
                    model="models/text-embedding-004",
                    google_api_key=self.api_key
                )
                logger.info("✅ Embedding model initialized")

            # If vectorstore exists, add to it; otherwise create new one
            if self.vectorstore is None:
                logger.info("🔄 Creating new vectorstore...")
                self.vectorstore = Chroma.from_texts(
                    texts=text_chunks,
                    embedding=self.embedding_model,
                    persist_directory=self.persist_path
                )
                logger.info(f"✅ Created new vectorstore with {len(text_chunks)} valid chunks.")
            else:
                logger.info("🔄 Adding to existing vectorstore...")
                self.vectorstore.add_texts(text_chunks)
                logger.info(f"✅ Added {len(text_chunks)} chunks to existing vectorstore.")

            # Persist the vectorstore
            self.vectorstore.persist()
            logger.info("💾 Vectorstore persisted successfully")

            # Initialize quiz generator after vectorstore is ready
            self.quiz_generator = QuizGenerator(self.llm, self.vectorstore)
            logger.info("🎯 Quiz generator initialized")

            # Test the vectorstore
            try:
                test_docs = self.vectorstore.similarity_search("test", k=1)
                logger.info(f"✅ Vectorstore test successful, found {len(test_docs)} documents")
            except Exception as e:
                logger.warning(f"⚠️ Vectorstore test failed: {e}")

            return True

        except Exception as e:
            logger.error(f"❌ Error setting up vectorstore: {e}")
            import traceback
            logger.error(f"❌ Traceback: {traceback.format_exc()}")
            return False

    def get_relevant_chunks(self, query: str, k: int = 10):
        """Retrieve relevant document chunks for the query."""
        if not self.vectorstore:
            logger.error("Vectorstore not initialized. Call setup_vectorstore first.")
            return []
        
        try:
            retriever = self.vectorstore.as_retriever(
                search_type="similarity", 
                search_kwargs={"k": k}
            )
            relevant_docs = retriever.get_relevant_documents(query)
            logger.info(f"✅ Retrieved {len(relevant_docs)} relevant chunks")
            return relevant_docs
        except Exception as e:
            logger.error(f"❌ Error retrieving chunks: {e}")
            return []
    
    def build_prompt(self, chunks, query: str) -> str:
        """Build the final prompt with context and query."""
        if not chunks:
            return f"""
You are a helpful AI assistant. The user asked: "{query}"

However, no relevant context was found in the document. Please let the user know that you cannot answer based on the provided context and suggest they try rephrasing their question or provide more specific details.

Question: {query}
Answer: """
        
        chunk_texts = [f"[Chunk {i+1}]: {chunk.page_content}" for i, chunk in enumerate(chunks)]
        context = "\n\n---\n\n".join(chunk_texts)
        
        return f"""
You are a helpful and smart AI assistant. Your goal is to answer the user's question using the following document context.

The answer should be grounded in the content — but you're allowed to infer, reason, and connect ideas **even if the answer is not stated explicitly**.

Use the context to support your reasoning. If the answer cannot reasonably be inferred or supported at all, then say: "I cannot answer this based on the provided context."

Here is the context:

{context}

Question: {query}

Answer: """
    
    def generate_response(self, prompt: str) -> str:
        """Generate response using Gemini."""
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            logger.error(f"❌ Error generating response: {e}")
            return "I apologize, but I encountered an error while generating the response. Please try again."
    
    def chat(self, query: str, k: int = 10) -> str:
        """Main chat interface - processes query and returns answer."""
        if not self.vectorstore:
            return "Please load a document first using the process_pdf method."
        
        try:
            # Retrieve relevant chunks
            relevant_chunks = self.get_relevant_chunks(query, k)
            
            # Build prompt
            final_prompt = self.build_prompt(relevant_chunks, query)
            
            # Generate and return response
            response = self.generate_response(final_prompt)
            logger.info(f"✅ Generated response for query: '{query[:50]}...'")
            return response
            
        except Exception as e:
            logger.error(f"❌ Error in chat method: {e}")
            return f"I encountered an error while processing your question: {str(e)}"
    


    def process_pdf(self, pdf_stream: io.BytesIO, chunk_size: int = 1000, chunk_overlap: int = 200) -> bool:
        """Complete pipeline to process a PDF file from an in-memory stream (Streamlit Cloud-safe)."""
        logger.info(f"🚀 Processing PDF from stream...")
        
        try:
            # Step 1: Parse PDF from BytesIO using PyMuPDF
            # Reset stream position to beginning
            pdf_stream.seek(0)
            
            # Read bytes from stream
            pdf_bytes = pdf_stream.read()
            
            # Open with PyMuPDF from bytes
            pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
            
            text = ""
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                page_text = page.get_text()
                if page_text:
                    text += page_text + "\n"
            
            pdf_document.close()
            
            if not text.strip():
                logger.error("❌ Failed to extract any text from PDF.")
                return False
            
            # Step 2: Create chunks (unchanged)
            chunks = self.create_chunks(text, chunk_size, chunk_overlap)
            if not chunks:
                logger.error("❌ No chunks created from extracted text.")
                return False
            
            # Step 3: Setup vectorstore (unchanged)
            success = self.setup_vectorstore(chunks)
            if success:
                # Since we don't have a filename in stream, you can skip or use a placeholder
                placeholder_name = f"streamed_file_{len(self.processed_files)+1}.pdf"
                if placeholder_name not in self.processed_files:
                    self.processed_files.append(placeholder_name)
                
                logger.info(f"✅ PDF processing completed successfully! Placeholder File: {placeholder_name}")
                return True
            else:
                logger.error("❌ Failed to setup vectorstore.")
                return False
            
        except Exception as e:
            logger.error(f"❌ Error processing PDF stream: {e}")
            return False

    
    # =========================
    # QUIZ FUNCTIONALITY
    # =========================
    
    def create_quiz(self, topic: str = "", num_questions: int = 5, 
                   difficulty: str = "medium", question_type: str = "multiple_choice") -> bool:
        """Create a new quiz based on the document content."""
        logger.info(f"🎯 Creating quiz with parameters: topic='{topic}', num_questions={num_questions}, difficulty={difficulty}, question_type={question_type}")
        
        if not self.quiz_generator:
            logger.error("❌ Quiz generator not initialized. Process a document first.")
            return False
        
        if not self.vectorstore:
            logger.error("❌ Vectorstore not initialized. Process a document first.")
            return False
        
        try:
            questions = self.quiz_generator.generate_quiz_questions(
                topic=topic,
                num_questions=num_questions,
                difficulty=difficulty,
                question_type=question_type
            )
            
            if not questions:
                logger.error("❌ No questions generated for quiz")
                return False
            
            self.quiz_manager.start_quiz(questions)
            logger.info(f"🎯 Quiz created successfully with {len(questions)} questions")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error creating quiz: {e}")
            import traceback
            logger.error(f"❌ Traceback: {traceback.format_exc()}")
            return False
    
    def get_current_quiz_question(self) -> Optional[Dict[str, Any]]:
        """Get the current quiz question."""
        question = self.quiz_manager.get_current_question()
        if question:
            return {
                'question': question.question,
                'options': question.options,
                'question_type': question.question_type,
                'difficulty': question.difficulty,
                'progress': self.quiz_manager.get_quiz_progress()
            }
        return None
    
    def submit_quiz_answer(self, answer: str) -> Dict[str, Any]:
        """Submit an answer to the current quiz question."""
        # Get current question before submitting answer
        current_question = self.quiz_manager.get_current_question()
        
        is_correct = self.quiz_manager.submit_answer(answer)
        progress = self.quiz_manager.get_quiz_progress()
        
        result = {
            'is_correct': is_correct,
            'progress': progress,
            'quiz_completed': progress['completed']
        }
        
        # Add correct answer and explanation
        if current_question:
            result['correct_answer'] = current_question.correct_answer
            result['explanation'] = current_question.explanation
        
        # If quiz is completed, calculate results
        if progress['completed']:
            quiz_result = self.quiz_manager.calculate_results()
            result['final_results'] = {
                'total_questions': quiz_result.total_questions,
                'correct_answers': quiz_result.correct_answers,
                'score_percentage': quiz_result.score_percentage,
                'time_taken': quiz_result.time_taken
            }
        
        return result
    
    def get_quiz_results(self) -> Optional[QuizResult]:
        """Get the results of the completed quiz."""
        if self.quiz_manager.get_quiz_progress()['completed']:
            return self.quiz_manager.calculate_results()
        return None
    
    def get_quiz_history(self) -> List[QuizResult]:
        """Get history of all quiz results."""
        return self.quiz_manager.get_quiz_history()
    
    def get_quiz_review(self) -> List[Dict[str, Any]]:
        """Get detailed review of the last quiz."""
        if not self.quiz_manager.user_answers:
            return []
        
        return self.quiz_manager.user_answers.copy()
    
    # =========================
    # EXISTING METHODS
    # =========================
    
    def get_processed_files(self) -> List[str]:
        """Get list of processed files."""
        return self.processed_files.copy()
    
    def clear_database(self) -> bool:
        """Clear the vector database."""
        try:
            if os.path.exists(self.persist_path):
                shutil.rmtree(self.persist_path)
                logger.info("🗑️ Cleared ChromaDB cache.")
            
            self.vectorstore = None
            self.quiz_generator = None
            self.processed_files = []
            return True
        except Exception as e:
            logger.error(f"❌ Error clearing database: {e}")
            return False
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        try:
            if hasattr(self, 'persist_path') and os.path.exists(self.persist_path):
                shutil.rmtree(self.persist_path)
                logger.info("🧹 Cleaned up temporary database on exit")
        except:
            pass  # Ignore cleanup errors



def main():
    """Example usage of the RAG Chatbot with Quiz functionality."""
    try:
        # Initialize chatbot
        chatbot = RAGChatbot()
        
        # Process PDF
        pdf_path = "iott.pdf"  # Make sure this file exists
        if not chatbot.process_pdf(pdf_path):
            print("Failed to process PDF. Exiting.")
            return
        
        # Interactive chat and quiz loop
        print("\n🤖 RAG Chatbot with Quiz ready!")
        print("Commands:")
        print("  'chat' - Chat with the document")
        print("  'quiz' - Take a quiz")
        print("  'history' - View quiz history")
        print("  'quit' - Exit")
        print("=" * 50)
        
        while True:
            command = input("\n📝 Enter command (chat/quiz/history/quit): ").strip().lower()
            
            if command in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            elif command == 'chat':
                # Chat mode
                while True:
                    query = input("\n💬 Your question (or 'back' to return): ").strip()
                    if query.lower() == 'back':
                        break
                    if not query:
                        print("Please enter a question.")
                        continue
                    
                    print("\n🤔 Thinking...")
                    answer = chatbot.chat(query)
                    print(f"\n🤖 Answer:\n{answer}")
                    print("-" * 50)
            
            elif command == 'quiz':
                # Quiz mode
                print("\n🎯 Quiz Configuration")
                
                # Get quiz parameters
                topic = input("Enter topic (or press Enter for general): ").strip()
                
                try:
                    num_questions = int(input("Number of questions (default 5): ") or "5")
                except ValueError:
                    num_questions = 5
                
                difficulty = input("Difficulty (easy/medium/hard, default medium): ").strip().lower()
                if difficulty not in ['easy', 'medium', 'hard']:
                    difficulty = 'medium'
                
                question_type = input("Question type (multiple_choice/true_false, default multiple_choice): ").strip().lower()
                if question_type not in ['multiple_choice', 'true_false']:
                    question_type = 'multiple_choice'
                
                print("\n🎲 Generating quiz...")
                
                # Create quiz
                if chatbot.create_quiz(topic, num_questions, difficulty, question_type):
                    print(f"✅ Quiz created successfully!")
                    
                    # Quiz loop
                    while True:
                        question_data = chatbot.get_current_quiz_question()
                        if not question_data:
                            break
                        
                        print(f"\n📊 Question {question_data['progress']['current_question']}/{question_data['progress']['total_questions']}")
                        print(f"🎯 Difficulty: {question_data['difficulty']}")
                        print(f"\n❓ {question_data['question']}")
                        
                        # Display options
                        for i, option in enumerate(question_data['options']):
                            print(f"   {option}")
                        
                        # Get user answer
                        if question_data['question_type'] == 'multiple_choice':
                            answer = input("\n📝 Your answer (A/B/C/D): ").strip().upper()
                        else:
                            answer = input("\n📝 Your answer (True/False): ").strip()
                        
                        # Submit answer
                        # Submit answer
                        result = chatbot.submit_quiz_answer(answer)

                        if result['is_correct']:
                            print("✅ Correct!")
                        else:
                            print("❌ Incorrect!")
                            if 'correct_answer' in result:
                                print(f"   Correct answer: {result['correct_answer']}")
                            if 'explanation' in result:
                                print(f"   Explanation: {result['explanation']}")
                        
                        if result['quiz_completed']:
                            # Show final results
                            final_results = result['final_results']
                            print("\n" + "="*50)
                            print("🎉 QUIZ COMPLETED!")
                            print(f"📊 Score: {final_results['correct_answers']}/{final_results['total_questions']}")
                            print(f"📈 Percentage: {final_results['score_percentage']:.1f}%")
                            print(f"⏱️ Time taken: {final_results['time_taken']}")
                            print("="*50)
                            
                            # Show review
                            review = chatbot.get_quiz_review()
                            if input("\n🔍 View detailed review? (y/n): ").lower() == 'y':
                                print("\n📋 QUIZ REVIEW:")
                                for i, item in enumerate(review, 1):
                                    status = "✅" if item['is_correct'] else "❌"
                                    print(f"\n{i}. {item['question']}")
                                    print(f"   Your answer: {item['user_answer']}")
                                    print(f"   Correct answer: {item['correct_answer']} {status}")
                                    print(f"   Explanation: {item['explanation']}")
                            break
                else:
                    print("❌ Failed to create quiz. Please try again.")
            
            elif command == 'history':
                # Show quiz history
                history = chatbot.get_quiz_history()
                if not history:
                    print("📭 No quiz history available.")
                else:
                    print("\n📈 QUIZ HISTORY:")
                    for i, result in enumerate(history, 1):
                        print(f"\n{i}. Quiz {i}")
                        print(f"   📊 Score: {result.correct_answers}/{result.total_questions}")
                        print(f"   📈 Percentage: {result.score_percentage:.1f}%")
                        print(f"   ⏱️ Time: {result.time_taken}")
            
            else:
                print("❌ Invalid command. Use 'chat', 'quiz', 'history', or 'quit'.")
    
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        logger.error(f"Application error: {e}")


if __name__ == "__main__":
    main()
