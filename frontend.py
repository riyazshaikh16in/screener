"""
Adaptive Hiring Screener - Streamlit Frontend
AI-powered mock interviewer based on Resume and JD
"""

import openai
import os
import json
from datetime import datetime
import streamlit as st
import PyPDF2
from docx import Document
from openai import OpenAI


st.set_page_config(
    page_title="AI Mock Interviewer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .score-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 0.5rem;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .score-value {
        font-size: 3rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .chat-container {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        max-height: 500px;
        overflow-y: auto;
    }
    .question-message {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #2196f3;
    }
    .answer-message {
        background-color: #f3e5f5;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #9c27b0;
    }
    .feedback-message {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)


def extract_pdf_text(pdf_file):
    """Extract text from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return None


def extract_docx_text(docx_file):
    """Extract text from DOCX file"""
    try:
        doc = Document(docx_file)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text
    except Exception as e:
        st.error(f"Error reading DOCX: {str(e)}")
        return None


def extract_text_from_file(uploaded_file):
    """Extract text from uploaded file (PDF or DOCX)"""
    if uploaded_file.type == "application/pdf":
        return extract_pdf_text(uploaded_file)
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return extract_docx_text(uploaded_file)
    else:
        st.error("Please upload a PDF or DOCX file")
        return None


def call_openai(prompt, system_prompt=None):
    """Call OpenAI API using new format"""
    try:
        if system_prompt is None:
            system_prompt = "You are an experienced technical interviewer conducting a professional interview."
        
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            st.error("❌ OpenAI API Key not configured")
            return None
            
        client = OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1500
        )
        return response.choices[0].message.content
    except Exception as e:
        raise Exception(f"OpenAI Error: {str(e)}")



def extract_resume_info(resume_text):
    """Extract key information from resume"""
    prompt = f"""
    Analyze this resume and extract key information in JSON format:
    
    RESUME:
    {resume_text[:2000]}
    
    Extract and return ONLY valid JSON with these fields:
    {{
        "name": "candidate name or Unknown",
        "experience_years": 0,
        "skills": ["skill1", "skill2"],
        "technologies": ["tech1", "tech2"],
        "companies": ["company1"],
        "roles": ["role1"],
        "education": "education details",
        "key_achievements": ["achievement1"],
        "summary": "brief professional summary"
    }}
    
    Return ONLY valid JSON, no additional text.
    """
    
    try:
        response = call_openai(prompt)
        # Clean the response
        response = response.strip()
        if response.startswith('```json'):
            response = response[7:]
        if response.startswith('```'):
            response = response[3:]
        if response.endswith('```'):
            response = response[:-3]
        response = response.strip()
        
        info = json.loads(response)
        return info
    except Exception as e:
        st.warning(f"Could not fully parse resume, but continuing: {str(e)}")
        return {
            "name": "Candidate",
            "experience_years": 0,
            "skills": [],
            "technologies": [],
            "companies": [],
            "roles": [],
            "education": "",
            "key_achievements": [],
            "summary": "Resume provided"
        }


def generate_interview_question(resume_info, jd_text=None, question_count=1, previous_questions=None):
    """Generate next interview question based on resume/JD"""
    
    if previous_questions is None:
        previous_questions = []
    
    previous_q_text = "\n".join([f"Q{i+1}: {q}" for i, q in enumerate(previous_questions[-3:])])  # Last 3 questions
    
    jd_section = f"\nJOB DESCRIPTION:\n{jd_text[:1000]}" if jd_text else ""
    
    prompt = f"""
    You are an experienced technical interviewer. Generate question #{question_count} for a mock interview.
    
    CANDIDATE INFO FROM RESUME:
    Name: {resume_info.get('name', 'Unknown')}
    Experience: {resume_info.get('experience_years', 0)} years
    Skills: {', '.join(resume_info.get('skills', [])[:5])}
    Companies: {', '.join(resume_info.get('companies', [])[:3])}
    Roles: {', '.join(resume_info.get('roles', [])[:3])}
    {jd_section}
    
    RECENT QUESTIONS (avoid repeating):
    {previous_q_text if previous_q_text else "None - this is the first question"}
    
    Guidelines:
    1. Ask ONE clear, specific question
    2. Base on their actual experience, skills, projects
    3. Progress from general to specific
    4. Make it conversational and natural
    5. Avoid repeating previous questions
    
    Return ONLY the question, nothing else.
    """
    
    try:
        question = call_openai(prompt, system_prompt="You are an expert technical recruiter and interviewer.")
        return question.strip()
    except Exception as e:
        st.error(f"Error generating question: {str(e)}")
        return "Tell me about your most recent project and what technologies you used."


def evaluate_answer(question, answer, resume_info, jd_text=None):
    """Evaluate candidate's answer"""
    
    jd_section = f"\nJOB DESCRIPTION:\n{jd_text[:500]}" if jd_text else ""
    
    prompt = f"""
    Evaluate this interview answer on a scale of 0-100.
    
    QUESTION: {question}
    
    ANSWER: {answer[:1000]}
    
    Provide evaluation in JSON format:
    {{
        "score": 75,
        "strengths": ["clear explanation"],
        "areas_for_improvement": ["more detail"],
        "feedback": "Good response"
    }}
    
    Return ONLY valid JSON, no additional text.
    """
    
    try:
        response = call_openai(prompt, system_prompt="You are an expert technical interviewer.")
        # Clean response
        response = response.strip()
        if response.startswith('```json'):
            response = response[7:]
        if response.startswith('```'):
            response = response[3:]
        if response.endswith('```'):
            response = response[:-3]
        response = response.strip()
        
        evaluation = json.loads(response)
        return evaluation
    except Exception as e:
        return {
            "score": 70,
            "strengths": ["Good effort"],
            "areas_for_improvement": ["Provide more detail"],
            "feedback": "Your answer shows understanding."
        }


def calculate_final_score(evaluations):
    """Calculate final interview score"""
    if not evaluations:
        return 0
    
    total = sum([e.get("score", 50) for e in evaluations])
    average = total / len(evaluations)
    return round(average, 1)


def display_chat_history():
    """Display chat history in interview"""
    if "chat_history" in st.session_state and st.session_state.chat_history:
        st.markdown("### 💬 Interview Chat History")
        
        chat_html = '<div class="chat-container">'
        
        for message in st.session_state.chat_history:
            if message["type"] == "question":
                chat_html += f'''
                <div class="question-message">
                    <strong>🤔 Interviewer:</strong><br>
                    {message["content"]}
                </div>
                '''
            elif message["type"] == "answer":
                chat_html += f'''
                <div class="answer-message">
                    <strong>👤 You:</strong><br>
                    {message["content"][:500]}...
                </div>
                '''
            elif message["type"] == "feedback":
                chat_html += f'''
                <div class="feedback-message">
                    <strong>📊 Score: {message.get("score", 0)}/100</strong><br>
                    {message["content"][:300]}...
                </div>
                '''
        
        chat_html += '</div>'
        st.markdown(chat_html, unsafe_allow_html=True)


def main():
    st.title("🎯 AI Mock Interviewer")
    st.markdown("Practice interviews with AI based on your resume and job description")
    
    # Initialize session state
    if "interview_stage" not in st.session_state:
        st.session_state.interview_stage = "setup"
    if "resume_text" not in st.session_state:
        st.session_state.resume_text = None
    if "jd_text" not in st.session_state:
        st.session_state.jd_text = None
    if "resume_info" not in st.session_state:
        st.session_state.resume_info = None
    if "start_from" not in st.session_state:
        st.session_state.start_from = None
    if "current_question" not in st.session_state:
        st.session_state.current_question = None
    if "question_count" not in st.session_state:
        st.session_state.question_count = 0
    if "asked_questions" not in st.session_state:
        st.session_state.asked_questions = []
    if "evaluations" not in st.session_state:
        st.session_state.evaluations = []
    if "final_score" not in st.session_state:
        st.session_state.final_score = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Sidebar - API Key
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.markdown("### OpenAI API Key")
        api_key = st.text_input(
            "Enter your OpenAI API Key",
            type="password",
            help="Your API key is not stored",
            placeholder="sk-..."
        )
        
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            openai.api_key = api_key
            st.success("✅ API Key configured")
        else:
            st.warning("⚠️ Please enter OpenAI API Key")
    
    # Setup Stage
    if st.session_state.interview_stage == "setup":
        st.header("📋 Setup Your Interview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📄 Upload Your Resume")
            resume_file = st.file_uploader(
                "Upload Resume (PDF or DOCX)",
                type=["pdf", "docx"],
                key="resume_uploader"
            )
            
            if resume_file:
                with st.spinner("📖 Reading resume..."):
                    resume_text = extract_text_from_file(resume_file)
                    if resume_text:
                        st.session_state.resume_text = resume_text
                        st.success("✅ Resume uploaded successfully")
                        
                        with st.spinner("🔍 Analyzing resume..."):
                            resume_info = extract_resume_info(resume_text)
                            if resume_info:
                                st.session_state.resume_info = resume_info
                                st.info(f"👤 Candidate: {resume_info.get('name', 'Unknown')}")
                                st.info(f"💼 Experience: {resume_info.get('experience_years', 'N/A')} years")
        
        with col2:
            st.subheader("📝 Upload Job Description (Optional)")
            jd_file = st.file_uploader(
                "Upload JD (PDF or DOCX)",
                type=["pdf", "docx"],
                key="jd_uploader"
            )
            
            if jd_file:
                with st.spinner("📖 Reading JD..."):
                    jd_text = extract_text_from_file(jd_file)
                    if jd_text:
                        st.session_state.jd_text = jd_text
                        st.success("✅ Job description uploaded successfully")
        
        st.divider()
        
        # Start Interview Section
        if st.session_state.resume_text and st.session_state.resume_info:
            st.subheader("🚀 Start Your Interview")
            
            if st.session_state.jd_text:
                st.markdown("**Choose where to start:**")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("📄 Start from Resume", use_container_width=True, type="primary"):
                        st.session_state.start_from = "resume"
                        st.session_state.interview_stage = "interviewing"
                        st.session_state.chat_history = []
                        st.session_state.current_question = None
                        st.session_state.question_count = 0
                        st.rerun()
                
                with col2:
                    if st.button("📋 Start from JD", use_container_width=True, type="primary"):
                        st.session_state.start_from = "jd"
                        st.session_state.interview_stage = "interviewing"
                        st.session_state.chat_history = []
                        st.session_state.current_question = None
                        st.session_state.question_count = 0
                        st.rerun()
                
                with col3:
                    if st.button("⚖️ Balanced Approach", use_container_width=True, type="primary"):
                        st.session_state.start_from = "balanced"
                        st.session_state.interview_stage = "interviewing"
                        st.session_state.chat_history = []
                        st.session_state.current_question = None
                        st.session_state.question_count = 0
                        st.rerun()
            else:
                if st.button("🎤 Start Mock Interview", use_container_width=True, type="primary"):
                    st.session_state.start_from = "resume"
                    st.session_state.interview_stage = "interviewing"
                    st.session_state.chat_history = []
                    st.session_state.current_question = None
                    st.session_state.question_count = 0
                    st.rerun()
        else:
            st.warning("⚠️ Please upload your resume to start the interview. Resume analysis is in progress...")
    
    # Interview Stage
    elif st.session_state.interview_stage == "interviewing":
        st.header("🎤 Mock Interview in Progress")
        st.info(f"📌 Interview Mode: **{st.session_state.start_from.upper()}**")
        
        # Generate first question if needed
        if st.session_state.current_question is None:
            with st.spinner("🤔 Generating first question..."):
                question = generate_interview_question(
                    st.session_state.resume_info,
                    st.session_state.jd_text,
                    question_count=1,
                    previous_questions=st.session_state.asked_questions
                )
                if question:
                    st.session_state.current_question = question
                    st.session_state.question_count = 1
                    st.session_state.chat_history.append({
                        "type": "question",
                        "content": question
                    })
                else:
                    st.error("Failed to generate question")
        
        # Display Chat History
        if st.session_state.chat_history:
            display_chat_history()
            st.divider()
        
        # Current Question
        st.markdown(f"### ❓ Question #{st.session_state.question_count}")
        st.markdown(f"**{st.session_state.current_question}**")
        st.divider()
        
        # Answer Input
        answer = st.text_area(
            "Your Answer:",
            height=150,
            placeholder="Type your answer here...",
            key=f"answer_{st.session_state.question_count}"
        )
        
        st.divider()
        
        # Control Buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            submit_answer = st.button("✅ Submit Answer", use_container_width=True, type="primary")
        
        with col2:
            skip_question = st.button("⏭️ Skip Question", use_container_width=True)
        
        with col3:
            stop_interview = st.button("🛑 Stop Interview", use_container_width=True, type="secondary")
        
        # Handle Submissions
        if submit_answer:
            if answer.strip():
                with st.spinner("📊 Evaluating your answer..."):
                    evaluation = evaluate_answer(
                        st.session_state.current_question,
                        answer,
                        st.session_state.resume_info,
                        st.session_state.jd_text
                    )
                    
                    st.session_state.evaluations.append(evaluation)
                    st.session_state.asked_questions.append(st.session_state.current_question)
                    
                    st.session_state.chat_history.append({
                        "type": "answer",
                        "content": answer
                    })
                    
                    feedback_text = f"**Strengths:** {', '.join(evaluation.get('strengths', []))}\n\n**Improvements:** {', '.join(evaluation.get('areas_for_improvement', []))}\n\n**Feedback:** {evaluation.get('feedback', 'Good response')}"
                    
                    st.session_state.chat_history.append({
                        "type": "feedback",
                        "content": feedback_text,
                        "score": evaluation.get('score', 50)
                    })
                
                st.success(f"✅ Evaluated! Score: {evaluation.get('score', 0)}/100")
                
                st.divider()
                if st.button("➡️ Next Question", use_container_width=True, type="primary"):
                    with st.spinner("🤔 Generating next question..."):
                        next_question = generate_interview_question(
                            st.session_state.resume_info,
                            st.session_state.jd_text,
                            question_count=st.session_state.question_count + 1,
                            previous_questions=st.session_state.asked_questions
                        )
                        if next_question:
                            st.session_state.current_question = next_question
                            st.session_state.question_count += 1
                            st.session_state.chat_history.append({
                                "type": "question",
                                "content": next_question
                            })
                            st.rerun()
            else:
                st.warning("⚠️ Please provide an answer")
        
        if skip_question:
            st.session_state.asked_questions.append(st.session_state.current_question)
            st.session_state.current_question = None
            st.rerun()
        
        if stop_interview:
            st.session_state.interview_stage = "completed"
            st.session_state.final_score = calculate_final_score(st.session_state.evaluations)
            st.rerun()
        
        # Progress
        st.divider()
        st.markdown("### 📊 Progress")
        st.progress(min(st.session_state.question_count / 10, 1.0))
        avg = calculate_final_score(st.session_state.evaluations)
        st.write(f"**Questions:** {len(st.session_state.asked_questions)} | **Average Score:** {avg:.1f}/100")
    
    # Completed Stage
    elif st.session_state.interview_stage == "completed":
        st.header("✅ Interview Completed!")
        st.divider()
        
        # Final Score
        st.markdown(f"""
        <div class="score-card">
            <h2>Your Final Score</h2>
            <div class="score-value">{st.session_state.final_score}</div>
            <p>out of 100</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Questions", len(st.session_state.asked_questions))
        
        with col2:
            if st.session_state.final_score >= 80:
                rating = "Excellent 🌟"
            elif st.session_state.final_score >= 70:
                rating = "Good 👍"
            elif st.session_state.final_score >= 60:
                rating = "Fair 📌"
            else:
                rating = "Improve 📚"
            st.metric("Rating", rating)
        
        with col3:
            st.metric("Best", max([e.get("score", 0) for e in st.session_state.evaluations], default=0))
        
        with col4:
            st.metric("Lowest", min([e.get("score", 100) for e in st.session_state.evaluations], default=0))
        
        st.divider()
        
        # Chat History
        st.markdown("### 💬 Complete Interview")
        display_chat_history()
        
        st.divider()
        
        # Detailed Feedback
        st.subheader("📝 Question-by-Question Feedback")
        
        for i, (q, e) in enumerate(zip(st.session_state.asked_questions, st.session_state.evaluations), 1):
            with st.expander(f"Q#{i} - Score: {e.get('score', 0)}/100"):
                st.write(f"**Question:** {q}")
                st.write(f"**Score:** {e.get('score', 0)}/100")
                st.write(f"**Strengths:** {', '.join(e.get('strengths', []))}")
                st.write(f"**Improve:** {', '.join(e.get('areas_for_improvement', []))}")
                st.write(f"**Feedback:** {e.get('feedback', '')}")
        
        st.divider()
        
        # Restart
        if st.button("🔄 Start New Interview", use_container_width=True, type="secondary"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


if __name__ == "__main__":
    main()
