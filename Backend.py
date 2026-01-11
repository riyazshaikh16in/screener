"""
Adaptive Hiring Screener - Backend
LLM-powered candidate evaluation system
"""
import os
import json
from datetime import datetime
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

evaluation_state = {
    "candidate_name": None,
    "job_title": None,
    "job_requirements": None,
    "resume_text": None,
    "assignment_response": None,
    "interview_answers": None,
    "resume_analysis": None,
    "assignment_eval": None,
    "interview_eval": None,
    "overall_recommendation": None,
    "confidence_level": None,
    "final_reasoning": None,
    "critical_red_flags": [],
    "follow_up_questions": [],
    "evaluation_path": [],
    "conversation_history": []
}


def initialize_evaluation_state(candidate_name, job_title, job_requirements, resume_text, assignment_response, interview_answers):
    global evaluation_state
    evaluation_state = {
        "candidate_name": candidate_name,
        "job_title": job_title,
        "job_requirements": job_requirements,
        "resume_text": resume_text,
        "assignment_response": assignment_response,
        "interview_answers": interview_answers,
        "resume_analysis": None,
        "assignment_eval": None,
        "interview_eval": None,
        "overall_recommendation": None,
        "confidence_level": None,
        "final_reasoning": None,
        "critical_red_flags": [],
        "follow_up_questions": [],
        "evaluation_path": [],
        "conversation_history": []
    }


def call_llm(prompt, system_prompt=None):
    """Call OpenAI LLM with given prompt"""
    try:
        if system_prompt is None:
            system_prompt = "You are an expert technical recruiter and hiring manager with deep experience in evaluating engineering talent. Provide thorough, fair, and constructive assessments."
        
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2000
        )
        return response.choices[0].message.content
    except Exception as e:
        raise Exception(f"LLM Error: {str(e)}")


def analyze_resume():
    """Analyze candidate's resume against job requirements"""
    global evaluation_state
    
    prompt = f"""
    Analyze this resume against the job requirements.
    
    JOB TITLE: {evaluation_state['job_title']}
    
    JOB REQUIREMENTS:
    {evaluation_state['job_requirements']}
    
    RESUME:
    {evaluation_state['resume_text']}
    
    Provide a detailed analysis in JSON format with the following fields:
    {{
        "score": (0-100),
        "experience_years": (number),
        "education_level": (string),
        "meets_minimum_requirements": (true/false),
        "skills": [list of identified skills],
        "relevant_experience": [list of relevant work experience],
        "gaps_or_concerns": [list of gaps or concerns],
        "reasoning": "detailed explanation"
    }}
    
    Return ONLY valid JSON, no additional text.
    """
    
    try:
        response = call_llm(prompt)
        analysis = json.loads(response)
        evaluation_state["resume_analysis"] = analysis
        evaluation_state["evaluation_path"].append("Resume analyzed for fit with job requirements")
        return analysis
    except Exception as e:
        raise Exception(f"Resume Analysis Error: {str(e)}")


def evaluate_assignment():
    """Evaluate technical assignment"""
    global evaluation_state
    
    prompt = f"""
    Evaluate this technical assignment submission.
    
    JOB TITLE: {evaluation_state['job_title']}
    
    JOB REQUIREMENTS:
    {evaluation_state['job_requirements']}
    
    ASSIGNMENT RESPONSE:
    {evaluation_state['assignment_response']}
    
    Provide a detailed evaluation in JSON format with the following fields:
    {{
        "score": (0-100),
        "correctness_score": (0-100),
        "code_quality_score": (0-100),
        "approach_score": (0-100),
        "problem_solving_score": (0-100),
        "technical_depth_score": (0-100),
        "strengths": [list of strengths],
        "weaknesses": [list of weaknesses],
        "red_flags": [list of red flags],
        "reasoning": "detailed explanation"
    }}
    
    Return ONLY valid JSON, no additional text.
    """
    
    try:
        response = call_llm(prompt)
        evaluation = json.loads(response)
        evaluation_state["assignment_eval"] = evaluation
        evaluation_state["evaluation_path"].append("Technical assignment evaluated for correctness and approach")
        return evaluation
    except Exception as e:
        raise Exception(f"Assignment Evaluation Error: {str(e)}")


def evaluate_interview():
    """Evaluate interview responses"""
    global evaluation_state
    
    interview_text = "\n".join([f"Q: {q}\nA: {a}" for q, a in evaluation_state["interview_answers"].items()])
    
    prompt = f"""
    Evaluate these interview responses.
    
    JOB TITLE: {evaluation_state['job_title']}
    
    JOB REQUIREMENTS:
    {evaluation_state['job_requirements']}
    
    INTERVIEW RESPONSES:
    {interview_text}
    
    Provide a detailed evaluation in JSON format with the following fields:
    {{
        "score": (0-100),
        "communication_score": (0-100),
        "cultural_fit_score": (0-100),
        "technical_depth_score": (0-100),
        "problem_solving_score": (0-100),
        "leadership_score": (0-100),
        "strengths": [list of strengths],
        "weaknesses": [list of weaknesses],
        "red_flags": [list of red flags],
        "reasoning": "detailed explanation"
    }}
    
    Return ONLY valid JSON, no additional text.
    """
    
    try:
        response = call_llm(prompt)
        evaluation = json.loads(response)
        evaluation_state["interview_eval"] = evaluation
        evaluation_state["evaluation_path"].append("Interview responses evaluated for communication and fit")
        return evaluation
    except Exception as e:
        raise Exception(f"Interview Evaluation Error: {str(e)}")


def generate_final_recommendation():
    """Generate final recommendation based on all evaluations"""
    global evaluation_state
    
    resume_score = evaluation_state["resume_analysis"].get("score", 0) if evaluation_state["resume_analysis"] else 0
    assignment_score = evaluation_state["assignment_eval"].get("score", 0) if evaluation_state["assignment_eval"] else 0
    interview_score = evaluation_state["interview_eval"].get("score", 0) if evaluation_state["interview_eval"] else 0
    
    prompt = f"""
    Make a final hiring recommendation based on all evaluation data.
    
    RESUME SCORE: {resume_score}/100
    ASSIGNMENT SCORE: {assignment_score}/100
    INTERVIEW SCORE: {interview_score}/100
    
    RESUME ANALYSIS:
    {json.dumps(evaluation_state["resume_analysis"], indent=2)}
    
    ASSIGNMENT EVALUATION:
    {json.dumps(evaluation_state["assignment_eval"], indent=2)}
    
    INTERVIEW EVALUATION:
    {json.dumps(evaluation_state["interview_eval"], indent=2)}
    
    Provide final recommendation in JSON format with the following fields:
    {{
        "overall_recommendation": ("HIRE" / "CONSIDER" / "REJECT"),
        "confidence_level": (0-100),
        "final_reasoning": "detailed explanation",
        "critical_red_flags": [list of critical red flags if any],
        "follow_up_questions": [list of follow-up questions to clarify weak areas]
    }}
    
    Return ONLY valid JSON, no additional text.
    """
    
    try:
        response = call_llm(prompt)
        recommendation = json.loads(response)
        
        evaluation_state["overall_recommendation"] = recommendation.get("overall_recommendation", "CONSIDER")
        evaluation_state["confidence_level"] = recommendation.get("confidence_level", 50)
        evaluation_state["final_reasoning"] = recommendation.get("final_reasoning", "")
        evaluation_state["critical_red_flags"] = recommendation.get("critical_red_flags", [])
        evaluation_state["follow_up_questions"] = recommendation.get("follow_up_questions", [])
        
        evaluation_state["evaluation_path"].append(f"Final recommendation: {evaluation_state['overall_recommendation']} (Confidence: {evaluation_state['confidence_level']}%)")
        
        return recommendation
    except Exception as e:
        raise Exception(f"Final Recommendation Error: {str(e)}")


def screen_candidate(candidate_name, job_title, job_requirements, resume_text, assignment_response, interview_answers):
    """Main function to screen a candidate"""
    try:
        initialize_evaluation_state(
            candidate_name=candidate_name,
            job_title=job_title,
            job_requirements=job_requirements,
            resume_text=resume_text,
            assignment_response=assignment_response,
            interview_answers=interview_answers
        )
        
        evaluate_assignment()
        evaluate_interview()
        analyze_resume()
        generate_final_recommendation()
        
        return evaluation_state
    except Exception as e:
        raise Exception(f"Screening Error: {str(e)}")


def get_evaluation_summary(state):
    """Convert evaluation state to summary format"""
    return {
        "candidate_name": state.get("candidate_name", "Unknown"),
        "job_title": state.get("job_title", "Unknown"),
        "submission_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "scores": {
            "resume": state.get("resume_analysis", {}).get("score", 0),
            "assignment": state.get("assignment_eval", {}).get("score", 0),
            "interview": state.get("interview_eval", {}).get("score", 0)
        },
        "overall_recommendation": state.get("overall_recommendation", "CONSIDER"),
        "confidence_level": state.get("confidence_level", 0),
        "final_reasoning": state.get("final_reasoning", ""),
        "resume_analysis": state.get("resume_analysis", {}),
        "assignment_eval": state.get("assignment_eval", {}),
        "interview_eval": state.get("interview_eval", {}),
        "critical_red_flags": state.get("critical_red_flags", []),
        "follow_up_questions": state.get("follow_up_questions", []),
        "evaluation_path": state.get("evaluation_path", [])
    }
