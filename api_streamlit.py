import streamlit as st
import requests
import json
import re
from datetime import datetime
import time

# Configure page
st.set_page_config(
    page_title="AI Education System",
    page_icon="📚",
    layout="wide"
)

# API Base URL
API_BASE_URL = "http://localhost:8000/api"

# Initialize session state
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'user_data' not in st.session_state:
    st.session_state.user_data = None
if 'current_step' not in st.session_state:
    st.session_state.current_step = 'login'
if 'assessment_data' not in st.session_state:
    st.session_state.assessment_data = None
if 'material_data' not in st.session_state:
    st.session_state.material_data = None
if 'quiz_data' not in st.session_state:
    st.session_state.quiz_data = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def extract_json_from_response(response_text):
    """Extract JSON from response that might contain text + JSON"""
    try:
        # First try to parse as direct JSON
        return json.loads(response_text)
    except json.JSONDecodeError:
        # If that fails, try to extract JSON from the text
        try:
            # Look for JSON block in the response
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            
            # Look for JSON object starting with { and ending with }
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            
            # If still no match, try to find where JSON starts
            lines = response_text.split('\n')
            json_started = False
            json_lines = []
            
            for line in lines:
                if line.strip().startswith('{'):
                    json_started = True
                if json_started:
                    json_lines.append(line)
                if json_started and line.strip().endswith('}'):
                    break
            
            if json_lines:
                json_text = '\n'.join(json_lines)
                return json.loads(json_text)
                
        except json.JSONDecodeError as e:
            st.error(f"Failed to parse JSON: {e}")
            st.code(response_text)  # Show the raw response for debugging
            return None
    
    return None

def make_api_request(endpoint, method="GET", data=None):
    """Make API request with error handling"""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            response = requests.post(url, json=data)
        
        if response.status_code == 200:
            # Try to parse as JSON first
            try:
                return response.json()
            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract JSON from text
                json_data = extract_json_from_response(response.text)
                return json_data
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Connection Error: {str(e)}")
        st.warning("Make sure the FastAPI server is running on localhost:8001")
        return None

def login_page():
    """User login and specialization selection"""
    st.title("🎓 AI Education System")
    st.subheader("Welcome to Personalized Learning!")
    
    with st.form("login_form"):
        st.write("### Get Started")
        username = st.text_input("Enter your name:", placeholder="e.g., John Doe")
        specialization = st.selectbox(
            "Choose your specialization:",
            ["data_engineer", "data_scientist"],
            format_func=lambda x: "Data Engineer 💻" if x == "data_engineer" else "Data Scientist 📊"
        )
        
        submitted = st.form_submit_button("Start Learning Journey", type="primary")
        
        if submitted and username:
            with st.spinner("Setting up your profile..."):
                response = make_api_request("/user/login", "POST", {
                    "username": username,
                    "specialization": specialization
                })
                
                if response:
                    st.session_state.user_id = response['user_id']
                    st.session_state.user_data = {
                        'username': username,
                        'specialization': specialization
                    }
                    st.session_state.current_step = 'assessment'
                    st.success(f"Welcome {username}! Ready to start your journey?")
                    time.sleep(1)
                    st.rerun()

def assessment_page():
    """Assessment page to determine user level"""
    st.title("📝 Skills Assessment")
    st.write(f"Hi {st.session_state.user_data['username']}! Let's assess your current level.")
    
    spec_display = "Data Engineering" if st.session_state.user_data['specialization'] == 'data_engineer' else "Data Science"
    st.info(f"This assessment will help us determine your level in {spec_display}")
    
    if st.session_state.assessment_data is None:
        if st.button("Start Assessment", type="primary"):
            with st.spinner("Generating personalized assessment questions..."):
                response = make_api_request(f"/assessment/{st.session_state.user_id}")
                
                # Debug: Show raw response
                if response is None:
                    st.error("Failed to get assessment data")
                    return
                
                # Handle different response formats
                if isinstance(response, dict):
                    if 'questions' in response:
                        # Direct questions format
                        st.session_state.assessment_data = response
                    else:
                        # Wrapped format
                        st.session_state.assessment_data = response
                else:
                    st.error("Unexpected response format")
                    return
                
                st.rerun()
    else:
        assessment = st.session_state.assessment_data
        
        # Handle different assessment data structures
        questions = []
        if 'questions' in assessment:
            questions = assessment['questions']
            total_questions = len(questions)
        else:
            st.error("No questions found in assessment data")
            st.json(assessment)  # Debug: show the structure
            return
        
        st.write(f"**Total Questions:** {total_questions}")
        st.write("Answer all questions honestly to get the most accurate level assessment.")
        
        with st.form("assessment_form"):
            answers = []
            for i, question in enumerate(questions):
                st.write(f"**Question {i+1}:** {question['question']}")
                
                answer = st.radio(
                    f"Select your answer for question {i+1}:",
                    options=range(len(question['options'])),
                    format_func=lambda x, opts=question['options']: f"{chr(65+x)}. {opts[x]}",
                    key=f"q_{question['id']}"
                )
                
                answers.append({
                    "question_id": question['id'],
                    "answer": str(answer)
                })
                
                st.write("---")
            
            submitted = st.form_submit_button("Submit Assessment", type="primary")
            
            if submitted:
                with st.spinner("Evaluating your answers..."):
                    response = make_api_request("/assessment/submit", "POST", {
                        "user_id": st.session_state.user_id,
                        "answers": answers
                    })
                    
                    if response:
                        st.success(f"Assessment completed! Your level: {response['level'].title()}")
                        st.write(f"**Score:** {response['score']}%")
                        st.write(f"**Recommendation:** {response['recommendation']}")
                        
                        st.session_state.user_data['level'] = response['level']
                        st.session_state.current_step = 'learning'
                        
                        time.sleep(2)
                        st.rerun()

def learning_page():
    """Main learning page with material and chat"""
    st.title("📚 Your Personalized Learning Material")
    
    # Load learning material if not loaded
    if st.session_state.material_data is None:
        with st.spinner("Creating your personalized learning material..."):
            response = make_api_request(f"/learning/start/{st.session_state.user_id}")
            if response:
                st.session_state.material_data = response
    
    if st.session_state.material_data:
        material = st.session_state.material_data['material']
        
        # Display material info
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header(material['title'])
            st.write(material['subtitle'])
            
            # Learning objectives
            st.write("### 🎯 Learning Objectives")
            for obj in material['learning_objectives']:
                st.write(f"• {obj}")
            
            # Prerequisites
            if material.get('prerequisites'):
                st.write("### 📋 Prerequisites")
                for prereq in material['prerequisites']:
                    st.write(f"• {prereq}")
            
            # Content
            st.write("### 📖 Course Content")
            
            # Theory section
            if 'content' in material and 'theory' in material['content']:
                theory = material['content']['theory']
                
                with st.expander("📚 Theory Overview", expanded=True):
                    st.write(theory.get('overview', ''))
                    
                    if theory.get('key_concepts'):
                        st.write("**Key Concepts:**")
                        for concept in theory['key_concepts']:
                            st.write(f"• {concept}")
                
                # Practical examples
                if material['content'].get('practical_examples'):
                    with st.expander("💡 Practical Examples"):
                        for example in material['content']['practical_examples']:
                            st.write(f"**{example['title']}**")
                            st.write(example['description'])
                            if example.get('code_snippet'):
                                st.code(example['code_snippet'])
                            st.write(example['explanation'])
                            st.write("---")
                
                # Best practices
                if material['content'].get('best_practices'):
                    with st.expander("✅ Best Practices"):
                        for practice in material['content']['best_practices']:
                            st.write(f"• {practice}")
        
        with col2:
            st.write("### 🤖 AI Tutor Chat")
            st.write("Ask questions about the material!")
            
            # Chat interface
            chat_container = st.container()
            
            # Display chat history
            with chat_container:
                for chat in st.session_state.chat_history:
                    with st.chat_message("user"):
                        st.write(chat['question'])
                    with st.chat_message("assistant"):
                        st.write(chat['answer'])
            
            # Chat input
            user_message = st.text_input("Ask about the material:", key="chat_input")
            
            if st.button("Send", type="secondary") and user_message:
                with st.spinner("AI Tutor is thinking..."):
                    response = make_api_request("/chat", "POST", {
                        "user_id": st.session_state.user_id,
                        "material_id": material['id'],
                        "message": user_message
                    })
                    
                    if response:
                        st.session_state.chat_history.append({
                            'question': user_message,
                            'answer': response['response']
                        })
                        st.rerun()
        
        # Navigation
        st.write("---")
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col2:
            if st.button("Take Quiz", type="primary", use_container_width=True):
                st.session_state.current_step = 'quiz'
                st.rerun()

def quiz_page():
    """Quiz page"""
    st.title("📝 Knowledge Check Quiz")
    st.write("Test your understanding of the material!")
    
    if st.session_state.quiz_data is None:
        with st.spinner("Generating personalized quiz..."):
            material_id = st.session_state.material_data['material']['id']
            response = make_api_request(f"/quiz/generate/{material_id}")
            if response:
                st.session_state.quiz_data = response
                st.rerun()
    
    if st.session_state.quiz_data:
        quiz = st.session_state.quiz_data['quiz']
        
        st.info(f"**Max Score:** {quiz.get('max_score', 100)} points")
        
        with st.form("quiz_form"):
            answers = []
            
            # Multiple choice questions
            if quiz.get('multiple_choice'):
                st.write("## 🔤 Multiple Choice Questions")
                for question in quiz['multiple_choice']:
                    st.write(f"**{question['question']}**")
                    
                    answer = st.radio(
                        "Select your answer:",
                        options=range(len(question['options'])),
                        format_func=lambda x, opts=question['options']: f"{chr(65+x)}. {opts[x]}",
                        key=f"mc_{question['id']}"
                    )
                    
                    answers.append({
                        "question_id": question['id'],
                        "answer": str(answer)
                    })
                    st.write("---")
            
            # Practical questions
            if quiz.get('practical_questions'):
                st.write("## 🛠️ Practical Questions")
                for question in quiz['practical_questions']:
                    st.write(f"**{question['question']}**")
                    if question.get('scenario'):
                        st.write(f"*Scenario:* {question['scenario']}")
                    
                    answer = st.radio(
                        "Choose the best solution:",
                        options=range(len(question['options'])),
                        format_func=lambda x, opts=question['options']: f"{chr(65+x)}. {opts[x]}",
                        key=f"pq_{question['id']}"
                    )
                    
                    answers.append({
                        "question_id": question['id'],
                        "answer": str(answer)
                    })
                    st.write("---")
            
            # Coding challenge
            coding_answer = None
            if quiz.get('coding_challenge'):
                st.write("## 💻 Coding Challenge")
                challenge = quiz['coding_challenge']
                
                st.write(f"**Problem:** {challenge['problem_description']}")
                
                if challenge.get('requirements'):
                    st.write("**Requirements:**")
                    for req in challenge['requirements']:
                        st.write(f"• {req}")
                
                if challenge.get('sample_input') and challenge.get('sample_output'):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Sample Input:**")
                        st.code(challenge['sample_input'])
                    with col2:
                        st.write("**Sample Output:**")
                        st.code(challenge['sample_output'])
                
                coding_answer = st.text_area(
                    "Write your code solution:",
                    height=200,
                    placeholder="Write your code here..."
                )
            
            submitted = st.form_submit_button("Submit Quiz", type="primary")
            
            if submitted:
                with st.spinner("Evaluating your answers..."):
                    material_id = st.session_state.material_data['material']['id']
                    response = make_api_request("/quiz/submit", "POST", {
                        "user_id": st.session_state.user_id,
                        "material_id": material_id,
                        "answers": answers,
                        "coding_answer": coding_answer
                    })
                    
                    if response:
                        st.session_state.quiz_result = response
                        st.session_state.current_step = 'results'
                        st.rerun()

def results_page():
    """Enhanced Results and report page with comprehensive AI feedback"""
    st.title("📊 Your Learning Report")
    # Di results_page(), tambahkan debug
    if 'quiz_result' in st.session_state:
        result = st.session_state.quiz_result['result']
        breakdown = result['breakdown']
        
        # DEBUG: Cek apakah enhanced_feedback ada
        if breakdown.get('coding'):
            coding = breakdown['coding']
            print("DEBUG Coding keys:", coding.keys())
            if 'enhanced_feedback' in coding:
                print("✅ Enhanced feedback found!")
            else:
                print("❌ Enhanced feedback MISSING!")
        
        # Performance summary cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Score", f"{result['total_score']}/{result['max_score']}")
        with col2:
            st.metric("Percentage", f"{result['percentage']}%")
        with col3:
            st.metric("Performance", result['performance'])
        with col4:
            status = "✅ PASSED" if result['passed'] else "❌ NEEDS REVIEW"
            st.metric("Status", status)
        
        # Overall AI feedback with better formatting
        st.write("## 🤖 AI Feedback & Analysis")
        feedback_container = st.container()
        
        with feedback_container:
            if result['performance'] == "Excellent":
                st.success(f"🎉 **{result['overall_feedback']}**")
            elif result['performance'] == "Good":
                st.info(f"👍 **{result['overall_feedback']}**")
            elif result['performance'] == "Satisfactory":
                st.warning(f"📈 **{result['overall_feedback']}**")
            else:
                st.error(f"💪 **{result['overall_feedback']}**")
        
        # Enhanced Detailed Performance Analysis
        st.write("## 📊 Detailed Performance Breakdown")
        
        breakdown = result['breakdown']
        
        # Create tabs for better organization
        tab1, tab2, tab3 = st.tabs(["🧠 Conceptual", "🛠️ Practical", "💻 Coding"])
        
        # Multiple Choice Questions Feedback
        with tab1:
            if breakdown.get('multiple_choice'):
                mc = breakdown['multiple_choice']
                
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.metric("Score", f"{mc['score']}/{mc['max_score']}")
                    progress_value = mc['score'] / max(mc['max_score'], 1)
                    st.progress(progress_value)
                    
                    # Performance indicator with more detailed feedback
                    if progress_value >= 0.8:
                        st.success("🌟 Excellent conceptual understanding!")
                        st.info("**AI Analysis:** You have mastered the theoretical concepts well. Your understanding of fundamental principles is solid.")
                    elif progress_value >= 0.6:
                        st.info("👍 Good conceptual grasp")
                        st.info("**AI Analysis:** You understand most concepts but there are some areas that need reinforcement.")
                    else:
                        st.warning("📚 Needs conceptual strengthening")
                        st.warning("**AI Analysis:** Focus on reviewing the fundamental concepts. Consider re-reading the material and seeking additional explanations.")
                
                with col2:
                    st.write("### 📝 Question-by-Question AI Feedback")
                    
                    # Enhanced feedback display
                    for i, feedback in enumerate(mc.get('feedback', []), 1):
                        with st.expander(f"Question {i}: {feedback['question'][:80]}..."):
                            
                            if feedback['is_correct']:
                                st.success("✅ **Correct Answer!**")
                                st.write(f"**Your choice:** Option {chr(65 + int(feedback.get('user_answer', 0)))}")
                            else:
                                st.error("❌ **Incorrect Answer**")
                                st.write(f"**Your choice:** Option {chr(65 + int(feedback.get('user_answer', 0)))}")
                                st.write(f"**Correct answer:** Option {chr(65 + feedback['correct_answer'])}")
                            
                            # AI explanation
                            if feedback.get('explanation'):
                                st.info(f"**🤖 AI Explanation:** {feedback['explanation']}")
                            
                            # Additional AI insights based on topic
                            topic = feedback.get('topic', 'general')
                            if not feedback['is_correct']:
                                st.write("**💡 Study Tip:**")
                                if 'data' in topic.lower():
                                    st.write("Focus on understanding data structures and their applications.")
                                elif 'algorithm' in topic.lower():
                                    st.write("Practice algorithm implementation and analyze time complexity.")
                                else:
                                    st.write(f"Review materials related to {topic} and practice similar problems.")
            else:
                st.info("No multiple choice questions in this quiz.")
        
        # Practical Questions Feedback
        with tab2:
            if breakdown.get('practical'):
                practical = breakdown['practical']
                
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.metric("Score", f"{practical['score']}/{practical['max_score']}")
                    progress_value = practical['score'] / max(practical['max_score'], 1)
                    st.progress(progress_value)
                    
                    # AI Analysis for practical skills
                    if progress_value >= 0.8:
                        st.success("🔧 Excellent practical application!")
                        st.info("**AI Analysis:** You excel at applying concepts to real-world scenarios. Your problem-solving approach is methodical and effective.")
                    elif progress_value >= 0.6:
                        st.info("🛠️ Good practical skills")
                        st.info("**AI Analysis:** You can apply concepts practically but could benefit from more diverse scenario practice.")
                    else:
                        st.warning("📋 Need more hands-on practice")
                        st.warning("**AI Analysis:** Focus on practical exercises and case studies. Try to connect theory with real-world applications.")
                
                with col2:
                    st.write("### 🛠️ Practical Application Analysis")
                    
                    for i, feedback in enumerate(practical.get('feedback', []), 1):
                        with st.expander(f"Scenario {i}: {feedback['question'][:80]}..."):
                            
                            if feedback['is_correct']:
                                st.success("✅ **Optimal Solution!**")
                                st.write("🤖 **AI Assessment:** Your approach demonstrates good practical understanding.")
                            else:
                                st.error("❌ **Suboptimal Solution**")
                                st.write(f"**Your approach:** Option {chr(65 + int(feedback.get('user_answer', 0)))}")
                                st.write(f"**Better approach:** Option {chr(65 + feedback['correct_answer'])}")
                            
                            # Detailed AI feedback
                            if feedback.get('explanation'):
                                st.info(f"**🤖 Why this approach works better:** {feedback['explanation']}")
                            
                            # Additional practical tips
                            st.write("**💼 Professional Tip:**")
                            if not feedback['is_correct']:
                                st.write("In real projects, always consider scalability, maintainability, and efficiency when choosing solutions.")
            else:
                st.info("No practical questions in this quiz.")
        
        # Enhanced Coding Challenge Feedback - Updated Section
# Enhanced Coding Challenge Feedback Section untuk results_page()
# Ganti bagian "with tab3:" dalam function results_page()

        with tab3:
            if breakdown.get('coding'):
                coding = breakdown['coding']
                
                # Header dengan score yang lebih menarik
                st.markdown("### 💻 **AI Code Review & Analysis**")
                
                # Score visualization yang lebih baik
                col1, col2, col3 = st.columns([2, 2, 3])
                
                with col1:
                    score_percentage = (coding['score'] / coding['max_score']) * 100
                    st.metric(
                        label="Code Score", 
                        value=f"{coding['score']}/{coding['max_score']}", 
                        delta=f"{score_percentage:.1f}%"
                    )
                    
                    # Visual progress bar dengan warna berdasarkan score
                    if score_percentage >= 80:
                        st.success(f"🌟 Excellent ({score_percentage:.0f}%)")
                    elif score_percentage >= 70:
                        st.info(f"👍 Good ({score_percentage:.0f}%)")
                    elif score_percentage >= 60:
                        st.warning(f"⚠️ Fair ({score_percentage:.0f}%)")
                    else:
                        st.error(f"❌ Needs Work ({score_percentage:.0f}%)")
                    
                    st.progress(score_percentage / 100)
                
                with col2:
                    # Quick assessment badges
                    enhanced_feedback = coding.get('enhanced_feedback', {})
                    if enhanced_feedback:
                        correctness = enhanced_feedback.get('detailed_feedback', {}).get('correctness', {})
                        status = correctness.get('status', 'unknown').title()
                        
                        st.markdown("**Quick Assessment:**")
                        if status == "Excellent":
                            st.success(f"✅ {status}")
                        elif status == "Good":
                            st.info(f"👍 {status}")
                        elif status == "Fair":
                            st.warning(f"⚠️ {status}")
                        else:
                            st.error(f"❌ {status}")
                
                with col3:
                    # Overall recommendation
                    if score_percentage >= 80:
                        st.success("🎉 **Outstanding work!** Your code demonstrates excellent programming skills.")
                    elif score_percentage >= 70:
                        st.info("👨‍💻 **Solid coding!** Good foundation with room for optimization.")
                    elif score_percentage >= 60:
                        st.warning("📈 **Getting there!** Focus on code quality and best practices.")
                    else:
                        st.error("💪 **Keep practicing!** Review fundamentals and try again.")
                
                st.markdown("---")
                
                # Enhanced feedback analysis
                enhanced_feedback = coding.get('enhanced_feedback', {})
                
                if enhanced_feedback:
                    # Create tabs for different aspects of code review
                    review_tab1, review_tab2, review_tab3, review_tab4 = st.tabs([
                        "🎯 Correctness", "📏 Code Quality", "🔍 Analysis", "💡 Solutions"
                    ])
                    
                    # Tab 1: Correctness Analysis
                    with review_tab1:
                        detailed = enhanced_feedback.get('detailed_feedback', {})
                        correctness = detailed.get('correctness', {})
                        
                        col_a, col_b = st.columns([1, 2])
                        
                        with col_a:
                            correctness_score = correctness.get('score', 0)
                            correctness_max = 30  # Based on your scoring system
                            correctness_pct = (correctness_score / correctness_max) * 100
                            
                            st.metric("Correctness Score", f"{correctness_score}/30")
                            st.progress(correctness_pct / 100)
                            
                            status = correctness.get('status', 'unknown').title()
                            if correctness_pct >= 85:
                                st.success(f"🌟 {status}")
                            elif correctness_pct >= 70:
                                st.info(f"👍 {status}")
                            elif correctness_pct >= 50:
                                st.warning(f"⚠️ {status}")
                            else:
                                st.error(f"❌ {status}")
                        
                        with col_b:
                            st.markdown("**🤖 AI Analysis:**")
                            explanation = correctness.get('explanation', 'Code functionality assessed')
                            st.info(explanation)
                            
                            # Test Results Display
                            test_results = correctness.get('test_results', [])
                            if test_results:
                                st.markdown("**🧪 Test Case Results:**")
                                
                                passed_tests = sum(1 for test in test_results if test.get('passed', False))
                                total_tests = len(test_results)
                                
                                if passed_tests == total_tests:
                                    st.success(f"✅ All {total_tests} test cases passed!")
                                else:
                                    st.warning(f"⚠️ {passed_tests}/{total_tests} test cases passed")
                                
                                # Show individual test results in expandable format
                                for i, test in enumerate(test_results, 1):
                                    with st.expander(f"Test Case {i}: {test.get('test_case', 'N/A')}", 
                                                   expanded=not test.get('passed', False)):
                                        
                                        if test.get('passed', False):
                                            st.success("✅ **PASSED**")
                                        else:
                                            st.error("❌ **FAILED**")
                                        
                                        if test.get('student_output'):
                                            st.code(f"Your Output: {test.get('student_output')}", language="text")
                                        
                                        if test.get('explanation'):
                                            st.info(f"💭 **Explanation:** {test.get('explanation')}")
                    
                    # Tab 2: Code Quality Assessment
                    with review_tab2:
                        quality = detailed.get('code_quality', {})
                        quality_score = quality.get('score', 0)
                        quality_max = 15
                        
                        st.metric("Overall Code Quality", f"{quality_score}/15")
                        st.progress((quality_score / quality_max))
                        
                        # Quality aspects breakdown
                        aspects = quality.get('aspects', {})
                        
                        st.markdown("**📊 Quality Breakdown:**")
                        
                        for aspect_name, aspect_data in aspects.items():
                            aspect_score = aspect_data.get('score', 0)
                            aspect_max = 5
                            feedback = aspect_data.get('feedback', 'No feedback available')
                            
                            # Create expandable section for each aspect
                            with st.expander(f"{aspect_name.replace('_', ' ').title()} - {aspect_score}/5"):
                                
                                # Visual score representation
                                score_cols = st.columns(5)
                                for i in range(5):
                                    if i < aspect_score:
                                        score_cols[i].success("⭐")
                                    else:
                                        score_cols[i].empty()
                                
                                st.write("**Feedback:**")
                                
                                if aspect_score >= 4:
                                    st.success(feedback)
                                elif aspect_score >= 3:
                                    st.info(feedback)
                                elif aspect_score >= 2:
                                    st.warning(feedback)
                                else:
                                    st.error(feedback)
                                
                                # Specific suggestions based on aspect
                                if aspect_name == "readability" and aspect_score < 4:
                                    st.markdown("**💡 Improvement Tips:**")
                                    st.write("• Use more descriptive variable names")
                                    st.write("• Add comments for complex logic")
                                    st.write("• Follow consistent formatting")
                                
                                elif aspect_name == "efficiency" and aspect_score < 4:
                                    st.markdown("**💡 Optimization Tips:**")
                                    st.write("• Consider algorithm complexity")
                                    st.write("• Look for redundant operations")
                                    st.write("• Use appropriate data structures")
                                
                                elif aspect_name == "best_practices" and aspect_score < 4:
                                    st.markdown("**💡 Best Practice Tips:**")
                                    st.write("• Follow PEP 8 style guidelines")
                                    st.write("• Use proper error handling")
                                    st.write("• Implement input validation")
                        
                        # Requirements compliance
                        requirements = detailed.get('requirements_compliance', {})
                        if requirements:
                            st.markdown("---")
                            st.markdown("**✅ Requirements Analysis:**")
                            
                            req_score = requirements.get('score', 0)
                            req_max = 5
                            
                            col_req1, col_req2 = st.columns([1, 3])
                            
                            with col_req1:
                                st.metric("Requirements Met", f"{req_score}/5")
                                st.progress(req_score / req_max)
                            
                            with col_req2:
                                checked_requirements = requirements.get('checked_requirements', [])
                                for req in checked_requirements:
                                    if req.get('met', False):
                                        st.success(f"✅ {req.get('requirement', 'N/A')}")
                                    else:
                                        st.error(f"❌ {req.get('requirement', 'N/A')}")
                                    
                                    if req.get('explanation'):
                                        st.caption(req.get('explanation'))
                    
                    # Tab 3: Detailed Analysis & Comparison
                    with review_tab3:
                        comparison = enhanced_feedback.get('code_comparison', {})
                        insights = enhanced_feedback.get('learning_insights', {})
                        
                        if comparison:
                            st.markdown("**🔍 Code Analysis:**")
                            
                            approach_sim = comparison.get('approach_similarity', '')
                            if approach_sim:
                                st.info(f"**Your Approach:** {approach_sim}")
                            
                            # Alternative approaches
                            alternatives = comparison.get('alternative_approaches', [])
                            if alternatives:
                                st.markdown("**🔄 Alternative Approaches:**")
                                for i, alt in enumerate(alternatives, 1):
                                    st.write(f"{i}. {alt}")
                            
                            # Improvements needed
                            improvements = comparison.get('improvements_needed', [])
                            if improvements:
                                st.markdown("**⚠️ Areas for Improvement:**")
                                for imp in improvements:
                                    st.warning(f"• {imp}")
                        
                        if insights:
                            st.markdown("---")
                            st.markdown("**🧠 Learning Insights:**")
                            
                            # Create columns for better layout
                            insight_col1, insight_col2 = st.columns(2)
                            
                            with insight_col1:
                                understood = insights.get('concepts_understood', [])
                                if understood:
                                    st.markdown("**✅ Concepts Mastered:**")
                                    for concept in understood:
                                        st.success(f"✅ {concept}")
                            
                            with insight_col2:
                                to_review = insights.get('concepts_to_review', [])
                                if to_review:
                                    st.markdown("**📚 Focus Areas:**")
                                    for concept in to_review:
                                        st.warning(f"📚 {concept}")
                            
                            # Practice suggestions
                            suggestions = insights.get('next_practice_suggestions', [])
                            if suggestions:
                                st.markdown("**🎯 Practice Recommendations:**")
                                for i, suggestion in enumerate(suggestions, 1):
                                    st.info(f"{i}. {suggestion}")
                    
                    # Tab 4: Solutions & Examples
                    with review_tab4:
                        ideal_solution = enhanced_feedback.get('ideal_solution', {})
                        ideal_explanation = enhanced_feedback.get('ideal_solution_explanation', {})
                        
                        if ideal_solution and ideal_solution.get('code'):
                            st.markdown("**💡 Optimal Solution:**")
                            
                            # Solution explanation
                            solution_explanation = ideal_solution.get('explanation', 'Here is an optimal solution:')
                            st.info(solution_explanation)
                            
                            # Code display with copy button
                            solution_code = ideal_solution.get('code', '# No solution available')
                            st.code(solution_code, language='python')
                            
                            # Detailed explanation
                            if ideal_explanation:
                                st.markdown("---")
                                st.markdown("**🎓 Solution Breakdown:**")
                                
                                why_this = ideal_explanation.get('why_this_approach', '')
                                if why_this:
                                    st.success(f"**Why This Approach:** {why_this}")
                                
                                techniques = ideal_explanation.get('key_techniques', [])
                                if techniques:
                                    st.markdown("**🔧 Key Techniques Used:**")
                                    for technique in techniques:
                                        st.write(f"• {technique}")
                                
                                complexity = ideal_explanation.get('complexity_analysis', '')
                                if complexity:
                                    st.markdown("**⚡ Complexity Analysis:**")
                                    st.info(complexity)
                        else:
                            st.info("💭 Ideal solution is being processed. Check back soon!")
                    
                    # Personalized feedback and action items
                    st.markdown("---")
                    
                    # Personalized feedback
                    personal_feedback = enhanced_feedback.get('personalized_feedback', '')
                    if personal_feedback:
                        st.markdown("### 🎯 **Personalized AI Feedback**")
                        
                        # Make it more visually appealing based on score
                        if score_percentage >= 80:
                            st.success(f"🎉 **{personal_feedback}**")
                        elif score_percentage >= 70:
                            st.info(f"👍 **{personal_feedback}**")
                        elif score_percentage >= 60:
                            st.warning(f"📈 **{personal_feedback}**")
                        else:
                            st.error(f"💪 **{personal_feedback}**")
                    
                    # Actionable steps
                    steps = enhanced_feedback.get('actionable_steps', [])
                    if steps:
                        st.markdown("### 🚀 **Next Action Steps**")
                        
                        for i, step in enumerate(steps, 1):
                            priority = "🔥" if i <= 2 else "📝"
                            st.write(f"{priority} **{i}.** {step}")
                
                else:
                    # Fallback display for basic feedback
                    st.warning("⚠️ Enhanced AI feedback is not available. Showing basic analysis:")
                    
                    coding_feedback = coding.get('feedback', {})
                    
                    # Basic feedback display
                    with st.expander("📝 Basic Code Review", expanded=True):
                        
                        st.markdown("**Correctness:**")
                        correctness = coding_feedback.get('correctness', 'Code functionality assessed')
                        st.info(correctness)
                        
                        st.markdown("**Code Quality:**")
                        quality = coding_feedback.get('code_quality', 'Code quality evaluated')
                        st.info(quality)
                        
                        st.markdown("**Requirements:**")
                        requirements = coding_feedback.get('requirements_met', 'Requirements checked')
                        st.info(requirements)
                        
                        # Show strengths and improvements
                        if coding_feedback.get('strengths'):
                            st.markdown("**💪 Strengths:**")
                            for strength in coding_feedback['strengths']:
                                st.success(f"✅ {strength}")
                        
                        if coding_feedback.get('areas_for_improvement'):
                            st.markdown("**🎯 Areas for Improvement:**")
                            for area in coding_feedback['areas_for_improvement']:
                                st.warning(f"⚠️ {area}")
                        
                        if coding_feedback.get('suggestions'):
                            st.markdown("**💡 Suggestions:**")
                            for suggestion in coding_feedback['suggestions']:
                                st.info(f"🤖 {suggestion}")
            
            else:
                st.info("ℹ️ No coding challenge in this quiz.")
        
        # AI-Powered Learning Insights
        st.write("## 🧠 AI Learning Insights & Recommendations")
        
        # Enhanced learning pattern analysis
        mc_score = breakdown.get('multiple_choice', {}).get('score', 0)
        mc_max = breakdown.get('multiple_choice', {}).get('max_score', 1)
        practical_score = breakdown.get('practical', {}).get('score', 0)
        practical_max = breakdown.get('practical', {}).get('max_score', 1)
        coding_score = breakdown.get('coding', {}).get('score', 0)
        coding_max = breakdown.get('coding', {}).get('max_score', 1)
        
        mc_pct = (mc_score / mc_max) * 100 if mc_max > 0 else 0
        practical_pct = (practical_score / practical_max) * 100 if practical_max > 0 else 0
        coding_pct = (coding_score / coding_max) * 100 if coding_max > 0 else 0
        
        # AI learning style analysis
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("### 🎯 Learning Profile")
            scores = {
                "Theoretical": mc_pct,
                "Applied": practical_pct,
                "Technical": coding_pct
            }
            
            strongest_area = max(scores, key=scores.get)
            weakest_area = min(scores, key=scores.get)
            
            st.success(f"**Strength:** {strongest_area}")
            st.write(f"Score: {scores[strongest_area]:.1f}%")
            
            st.warning(f"**Focus Area:** {weakest_area}")
            st.write(f"Score: {scores[weakest_area]:.1f}%")
        
        with col2:
            st.write("### 🤖 AI Learning Style")
            
            # Determine learning style based on performance pattern
            if mc_pct > practical_pct and mc_pct > coding_pct:
                learning_style = "Theoretical Learner"
                style_desc = "You excel at understanding concepts and theory"
            elif practical_pct > mc_pct and practical_pct > coding_pct:
                learning_style = "Applied Learner"
                style_desc = "You learn best through practical applications"
            elif coding_pct > mc_pct and coding_pct > practical_pct:
                learning_style = "Hands-on Coder"
                style_desc = "You prefer learning through coding and implementation"
            else:
                learning_style = "Balanced Learner"
                style_desc = "You have a well-rounded learning approach"
            
            st.info(f"**Style:** {learning_style}")
            st.write(style_desc)
        
        with col3:
            st.write("### 📈 Progress Level")
            
            if result['percentage'] >= 85:
                level = "Advanced"
                next_step = "Ready for expert-level challenges"
            elif result['percentage'] >= 70:
                level = "Intermediate"
                next_step = "Moving toward advanced concepts"
            elif result['percentage'] >= 60:
                level = "Developing"
                next_step = "Building solid foundation"
            else:
                level = "Beginner"
                next_step = "Focus on fundamentals"
            
            st.info(f"**Current Level:** {level}")
            st.write(next_step)
        
        # Personalized AI Study Plan
        st.write("### 🎯 Personalized AI Study Plan")
        
        study_plan_container = st.container()
        with study_plan_container:
            
            # Immediate actions
            st.write("#### 🚀 Immediate Actions (This Week)")
            immediate_actions = []
            
            if not result['passed']:
                immediate_actions.append("📚 Review incorrect answers and understand the explanations")
                immediate_actions.append("🔄 Retake the quiz after reviewing weak areas")
            
            if mc_pct < 70:
                immediate_actions.append("📖 Study theoretical concepts more deeply")
                immediate_actions.append("💭 Create concept maps to visualize relationships")
            
            if practical_pct < 70:
                immediate_actions.append("🛠️ Practice with real-world scenarios and case studies")
                immediate_actions.append("🔍 Analyze how concepts apply in different contexts")
            
            if coding_pct < 70:
                immediate_actions.append("💻 Complete coding exercises daily (30-60 minutes)")
                immediate_actions.append("🐛 Focus on debugging and code improvement")
            
            if result['percentage'] >= 80:
                immediate_actions.append("🎉 Celebrate your achievement!")
                immediate_actions.append("🚀 Start exploring advanced topics")
            
            for action in immediate_actions:
                st.write(f"• {action}")
            
            # Short-term goals
            st.write("#### 📅 Short-term Goals (Next 2-4 Weeks)")
            short_term = []
            
            if weakest_area == "Theoretical":
                short_term.append("📚 Complete additional reading on fundamental concepts")
                short_term.append("🎓 Take supplementary online courses")
            elif weakest_area == "Applied":
                short_term.append("🏗️ Work on 2-3 practical projects")
                short_term.append("👥 Join study groups or discussion forums")
            elif weakest_area == "Technical":
                short_term.append("💻 Solve 20+ coding problems")
                short_term.append("🔧 Build a portfolio project")
            
            short_term.append("📊 Track daily learning progress")
            short_term.append("🎯 Set specific skill improvement targets")
            
            for goal in short_term:
                st.write(f"• {goal}")
            
            # Long-term development
            st.write("#### 🌟 Long-term Development (Next 2-3 Months)")
            long_term = []
            
            if result['percentage'] >= 80:
                long_term.append("🎓 Move to advanced specialization topics")
                long_term.append("🏆 Contribute to open source projects")
                long_term.append("👨‍🏫 Consider mentoring other learners")
            else:
                long_term.append("💪 Achieve consistent 80%+ scores on assessments")
                long_term.append("🔄 Regularly review and reinforce learned concepts")
                long_term.append("📈 Gradually increase complexity of practice problems")
            
            long_term.append("🌐 Stay updated with industry trends")
            long_term.append("🤝 Network with professionals in your field")
            
            for goal in long_term:
                st.write(f"• {goal}")
        
        # Generate comprehensive AI report
        if st.button("📋 Generate Full AI Report", type="primary"):
            with st.spinner("🤖 AI is analyzing your performance and generating comprehensive report..."):
                try:
                    result_id = st.session_state.quiz_result.get('result_id')
                    if result_id:
                        # Show enhanced features notification
                        st.info("🚀 **Enhanced Features Included:**\n" +
                               "• Detailed code analysis with test case results\n" +
                               "• Comparison with ideal solution\n" +
                               "• Personalized learning insights\n" +
                               "• Actionable improvement suggestions")
                        
                        response = make_api_request(f"/report/{result_id}")
                        
                        if response and 'report' in response:
                            display_comprehensive_ai_report(response['report'])
                        else:
                            st.error("Failed to generate AI report. Please try again.")
                    else:
                        st.error("No result ID found. Please retake the quiz.")
                except Exception as e:
                    st.error(f"Error generating report: {str(e)}")
        
        # Navigation with better UX
        st.write("---")
        st.write("### 🧭 What's Next?")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Improve & Retake", type="secondary", use_container_width=True):
                st.info("💡 **AI Suggestion:** Review the feedback above before retaking!")
                if st.button("Confirm Retake", type="primary"):
                    st.session_state.quiz_data = None
                    st.session_state.current_step = 'quiz'
                    st.rerun()
        
        with col2:
            if st.button("📚 New Learning Topic", type="secondary", use_container_width=True):
                st.session_state.material_data = None
                st.session_state.quiz_data = None
                st.session_state.chat_history = []
                st.session_state.current_step = 'learning'
                st.rerun()
        
        with col3:
            if st.button("🏠 Dashboard", type="secondary", use_container_width=True):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()

    else:
        st.error("❌ No quiz results found. Please complete a quiz first.")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📝 Take Quiz", type="primary"):
                st.session_state.current_step = 'quiz'
                st.rerun()
        with col2:
            if st.button("📚 Start Learning", type="secondary"):
                st.session_state.current_step = 'learning'
                st.rerun()

def display_comprehensive_ai_report(report):
    """Display the comprehensive AI-generated report"""
    st.write("## 📋 Comprehensive AI Learning Report")
    
    # Student info
    student_info = report.get('student_info', {})
    st.write(f"**👤 Student:** {student_info.get('name', 'Unknown')}")
    st.write(f"**🎯 Specialization:** {student_info.get('specialization', 'General')}")
    st.write(f"**📊 Current Level:** {student_info.get('current_level', 'Beginner')}")
    st.write(f"**📚 Material Completed:** {student_info.get('material_completed', 'N/A')}")
    
    # Performance summary
    performance = report.get('performance_summary', {})
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Overall Score", f"{performance.get('overall_score', 0)}%")
    with col2:
        st.metric("Performance Level", performance.get('performance_level', 'N/A'))
    with col3:
        status = "PASSED" if performance.get('passed', False) else "NEEDS REVIEW"
        st.metric("Status", status)
    
    # Achievements
    if report.get('achievements'):
        st.write("### 🏆 Achievements Unlocked")
        achievement_cols = st.columns(min(len(report['achievements']), 4))
        for i, achievement in enumerate(report['achievements']):
            col_idx = i % len(achievement_cols)
            with achievement_cols[col_idx]:
                st.write(f"{achievement.get('icon', '🏅')}")
                st.write(f"**{achievement.get('badge', 'Achievement')}**")
                st.caption(achievement.get('description', ''))
    
    # Learning insights
    if report.get('learning_insights'):
        insights = report['learning_insights']
        st.write("### 🧠 AI Learning Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Learning Style:** {insights.get('learning_style', 'balanced').title()}")
            st.info(f"**Strongest Area:** {insights.get('strength_area', 'N/A')}")
        
        with col2:
            st.warning(f"**Focus Area:** {insights.get('improvement_area', 'N/A')}")
            st.info(f"**Progress Trend:** {insights.get('progress_trend', 'steady').title()}")
        
        if insights.get('study_recommendations'):
            st.write("**🎯 Personalized Study Methods:**")
            for rec in insights['study_recommendations']:
                st.write(f"• {rec}")
    
# AI Feedback Analysis
    if report.get('ai_feedback'):
        ai_feedback = report['ai_feedback']
        st.write("### 🤖 Advanced AI Analysis")
        
        # Strengths identified by AI
        if ai_feedback.get('strengths'):
            st.write("#### 💪 AI-Identified Strengths")
            for strength in ai_feedback['strengths']:
                st.success(f"✅ {strength}")
        
        # Areas for improvement with AI suggestions
        if ai_feedback.get('improvement_areas'):
            st.write("#### 🎯 AI-Recommended Improvements")
            for area in ai_feedback['improvement_areas']:
                st.warning(f"⚠️ {area['issue']}")
                if area.get('suggestion'):
                    st.info(f"💡 AI Suggestion: {area['suggestion']}")
        
        # Coding-specific AI feedback
        if ai_feedback.get('coding_analysis'):
            coding_analysis = ai_feedback['coding_analysis']
            st.write("#### 💻 Advanced Code Analysis")
            
            with st.expander("🔍 Detailed Code Review", expanded=True):
                # Code complexity analysis
                if coding_analysis.get('complexity'):
                    complexity = coding_analysis['complexity']
                    st.write("**Complexity Analysis:**")
                    st.write(f"• Time Complexity: `{complexity.get('time', 'O(n)')}`")
                    st.write(f"• Space Complexity: `{complexity.get('space', 'O(1)')}`")
                    st.write(f"• AI Assessment: {complexity.get('assessment', 'Acceptable')}")
                
                # Code patterns detected
                if coding_analysis.get('patterns'):
                    st.write("**Code Patterns Detected:**")
                    for pattern in coding_analysis['patterns']:
                        if pattern.get('positive', True):
                            st.success(f"✅ Good use of: {pattern['name']}")
                        else:
                            st.warning(f"⚠️ Consider improving: {pattern['name']}")
                        st.caption(pattern.get('description', ''))
                
                # Code quality metrics
                if coding_analysis.get('quality_metrics'):
                    metrics = coding_analysis['quality_metrics']
                    st.write("**Code Quality Metrics:**")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        readability = metrics.get('readability', 0)
                        st.metric("Readability", f"{readability}/10")
                        if readability >= 8:
                            st.success("Excellent")
                        elif readability >= 6:
                            st.info("Good")
                        else:
                            st.warning("Needs work")
                    
                    with col2:
                        maintainability = metrics.get('maintainability', 0)
                        st.metric("Maintainability", f"{maintainability}/10")
                        if maintainability >= 8:
                            st.success("Excellent")
                        elif maintainability >= 6:
                            st.info("Good")
                        else:
                            st.warning("Needs work")
                    
                    with col3:
                        efficiency = metrics.get('efficiency', 0)
                        st.metric("Efficiency", f"{efficiency}/10")
                        if efficiency >= 8:
                            st.success("Excellent")
                        elif efficiency >= 6:
                            st.info("Good")
                        else:
                            st.warning("Needs work")
    
    # Personalized learning path
    if report.get('learning_path'):
        learning_path = report['learning_path']
        st.write("### 🛤️ AI-Generated Learning Path")
        
        # Current position
        current_position = learning_path.get('current_position', {})
        st.info(f"**Current Position:** {current_position.get('level', 'Beginner')} - {current_position.get('description', '')}")
        
        # Next steps with timeline
        if learning_path.get('next_steps'):
            st.write("#### 📅 Recommended Learning Timeline")
            
            for i, step in enumerate(learning_path['next_steps'], 1):
                with st.expander(f"Week {i}: {step.get('title', f'Step {i}')}"):
                    st.write(f"**Focus:** {step.get('focus', 'Practice and review')}")
                    st.write(f"**Duration:** {step.get('duration', '1 week')}")
                    
                    if step.get('activities'):
                        st.write("**Activities:**")
                        for activity in step['activities']:
                            st.write(f"• {activity}")
                    
                    if step.get('resources'):
                        st.write("**Recommended Resources:**")
                        for resource in step['resources']:
                            st.write(f"📚 {resource}")
                    
                    if step.get('success_criteria'):
                        st.write("**Success Criteria:**")
                        for criteria in step['success_criteria']:
                            st.write(f"🎯 {criteria}")
        
        # Long-term goals
        if learning_path.get('long_term_goals'):
            st.write("#### 🌟 Long-term Learning Goals")
            for goal in learning_path['long_term_goals']:
                st.write(f"🎯 **{goal.get('title', 'Goal')}** (Target: {goal.get('timeline', '3 months')})")
                st.write(f"   {goal.get('description', '')}")
    
    # Practice recommendations with difficulty progression
    if report.get('practice_recommendations'):
        practice = report['practice_recommendations']
        st.write("### 🎯 AI-Curated Practice Plan")
        
        # Immediate practice (this week)
        if practice.get('immediate'):
            st.write("#### 🚀 This Week's Practice")
            immediate_practice = practice['immediate']
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Coding Challenges:**")
                for challenge in immediate_practice.get('coding_challenges', []):
                    difficulty_color = {
                        'easy': '🟢',
                        'medium': '🟡', 
                        'hard': '🔴'
                    }.get(challenge.get('difficulty', 'easy').lower(), '🟢')
                    
                    st.write(f"{difficulty_color} {challenge.get('title', 'Practice Problem')}")
                    st.caption(f"Focus: {challenge.get('focus', 'General practice')}")
            
            with col2:
                st.write("**Concept Review:**")
                for concept in immediate_practice.get('concepts', []):
                    st.write(f"📚 {concept.get('topic', 'Topic')}")
                    st.caption(f"Priority: {concept.get('priority', 'Medium')}")
        
        # Progressive practice plan
        if practice.get('progressive'):
            st.write("#### 📈 Progressive Practice Plan")
            for week, plan in enumerate(practice['progressive'], 1):
                with st.expander(f"Week {week}: {plan.get('theme', f'Practice Week {week}')}"):
                    st.write(f"**Difficulty Level:** {plan.get('difficulty', 'Beginner')}")
                    st.write(f"**Focus Areas:** {', '.join(plan.get('focus_areas', ['General']))}")
                    
                    if plan.get('problems'):
                        st.write("**Recommended Problems:**")
                        for problem in plan['problems']:
                            st.write(f"• {problem}")
                    
                    if plan.get('time_allocation'):
                        st.write(f"**Suggested Time:** {plan['time_allocation']}")
# AI Feedback Analysis
    if report.get('ai_feedback'):
        ai_feedback = report['ai_feedback']
        st.write("### 🤖 Advanced AI Analysis")
        
        # Strengths identified by AI
        if ai_feedback.get('strengths'):
            st.write("#### 💪 AI-Identified Strengths")
            for strength in ai_feedback['strengths']:
                st.success(f"✅ {strength}")
        
        # Areas for improvement with AI suggestions
        if ai_feedback.get('improvement_areas'):
            st.write("#### 🎯 AI-Recommended Improvements")
            for area in ai_feedback['improvement_areas']:
                st.warning(f"⚠️ {area['issue']}")
                if area.get('suggestion'):
                    st.info(f"💡 AI Suggestion: {area['suggestion']}")
        
        # Coding-specific AI feedback
        if ai_feedback.get('coding_analysis'):
            coding_analysis = ai_feedback['coding_analysis']
            st.write("#### 💻 Advanced Code Analysis")
            
            with st.expander("🔍 Detailed Code Review", expanded=True):
                # Code complexity analysis
                if coding_analysis.get('complexity'):
                    complexity = coding_analysis['complexity']
                    st.write("**Complexity Analysis:**")
                    st.write(f"• Time Complexity: `{complexity.get('time', 'O(n)')}`")
                    st.write(f"• Space Complexity: `{complexity.get('space', 'O(1)')}`")
                    st.write(f"• AI Assessment: {complexity.get('assessment', 'Acceptable')}")
                
                # Code patterns detected
                if coding_analysis.get('patterns'):
                    st.write("**Code Patterns Detected:**")
                    for pattern in coding_analysis['patterns']:
                        if pattern.get('positive', True):
                            st.success(f"✅ Good use of: {pattern['name']}")
                        else:
                            st.warning(f"⚠️ Consider improving: {pattern['name']}")
                        st.caption(pattern.get('description', ''))
                
                # Code quality metrics
                if coding_analysis.get('quality_metrics'):
                    metrics = coding_analysis['quality_metrics']
                    st.write("**Code Quality Metrics:**")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        readability = metrics.get('readability', 0)
                        st.metric("Readability", f"{readability}/10")
                        if readability >= 8:
                            st.success("Excellent")
                        elif readability >= 6:
                            st.info("Good")
                        else:
                            st.warning("Needs work")
                    
                    with col2:
                        maintainability = metrics.get('maintainability', 0)
                        st.metric("Maintainability", f"{maintainability}/10")
                        if maintainability >= 8:
                            st.success("Excellent")
                        elif maintainability >= 6:
                            st.info("Good")
                        else:
                            st.warning("Needs work")
                    
                    with col3:
                        efficiency = metrics.get('efficiency', 0)
                        st.metric("Efficiency", f"{efficiency}/10")
                        if efficiency >= 8:
                            st.success("Excellent")
                        elif efficiency >= 6:
                            st.info("Good")
                        else:
                            st.warning("Needs work")
    
    # Personalized learning path
    if report.get('learning_path'):
        learning_path = report['learning_path']
        st.write("### 🛤️ AI-Generated Learning Path")
        
        # Current position
        current_position = learning_path.get('current_position', {})
        st.info(f"**Current Position:** {current_position.get('level', 'Beginner')} - {current_position.get('description', '')}")
        
        # Next steps with timeline
        if learning_path.get('next_steps'):
            st.write("#### 📅 Recommended Learning Timeline")
            
            for i, step in enumerate(learning_path['next_steps'], 1):
                with st.expander(f"Week {i}: {step.get('title', f'Step {i}')}"):
                    st.write(f"**Focus:** {step.get('focus', 'Practice and review')}")
                    st.write(f"**Duration:** {step.get('duration', '1 week')}")
                    
                    if step.get('activities'):
                        st.write("**Activities:**")
                        for activity in step['activities']:
                            st.write(f"• {activity}")
                    
                    if step.get('resources'):
                        st.write("**Recommended Resources:**")
                        for resource in step['resources']:
                            st.write(f"📚 {resource}")
                    
                    if step.get('success_criteria'):
                        st.write("**Success Criteria:**")
                        for criteria in step['success_criteria']:
                            st.write(f"🎯 {criteria}")
        
        # Long-term goals
        if learning_path.get('long_term_goals'):
            st.write("#### 🌟 Long-term Learning Goals")
            for goal in learning_path['long_term_goals']:
                st.write(f"🎯 **{goal.get('title', 'Goal')}** (Target: {goal.get('timeline', '3 months')})")
                st.write(f"   {goal.get('description', '')}")
    
    # Practice recommendations with difficulty progression
    if report.get('practice_recommendations'):
        practice = report['practice_recommendations']
        st.write("### 🎯 AI-Curated Practice Plan")
        
        # Immediate practice (this week)
        if practice.get('immediate'):
            st.write("#### 🚀 This Week's Practice")
            immediate_practice = practice['immediate']
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Coding Challenges:**")
                for challenge in immediate_practice.get('coding_challenges', []):
                    difficulty_color = {
                        'easy': '🟢',
                        'medium': '🟡', 
                        'hard': '🔴'
                    }.get(challenge.get('difficulty', 'easy').lower(), '🟢')
                    
                    st.write(f"{difficulty_color} {challenge.get('title', 'Practice Problem')}")
                    st.caption(f"Focus: {challenge.get('focus', 'General practice')}")
            
            with col2:
                st.write("**Concept Review:**")
                for concept in immediate_practice.get('concepts', []):
                    st.write(f"📚 {concept.get('topic', 'Topic')}")
                    st.caption(f"Priority: {concept.get('priority', 'Medium')}")
        
        # Progressive practice plan
        if practice.get('progressive'):
            st.write("#### 📈 Progressive Practice Plan")
            for week, plan in enumerate(practice['progressive'], 1):
                with st.expander(f"Week {week}: {plan.get('theme', f'Practice Week {week}')}"):
                    st.write(f"**Difficulty Level:** {plan.get('difficulty', 'Beginner')}")
                    st.write(f"**Focus Areas:** {', '.join(plan.get('focus_areas', ['General']))}")
                    
                    if plan.get('problems'):
                        st.write("**Recommended Problems:**")
                        for problem in plan['problems']:
                            st.write(f"• {problem}")
                    
                    if plan.get('time_allocation'):
                        st.write(f"**Suggested Time:** {plan['time_allocation']}")
    
    # AI Mentor Messages
    if report.get('ai_mentor_message'):
        st.write("### 🤖 Personal Message from AI Mentor")
        mentor_message = report['ai_mentor_message']
        
        # Personalized encouragement
        if mentor_message.get('encouragement'):
            st.success(f"🌟 **Encouragement:** {mentor_message['encouragement']}")
        
        # Specific advice
        if mentor_message.get('advice'):
            st.info(f"💡 **Advice:** {mentor_message['advice']}")
        
        # Challenge for next level
        if mentor_message.get('challenge'):
            st.warning(f"🎯 **Challenge:** {mentor_message['challenge']}")
        
        # Motivational quote or tip
        if mentor_message.get('motivation'):
            st.write(f"✨ *{mentor_message['motivation']}*")
    
    # Report generation timestamp and metadata
    st.write("---")
    report_meta = report.get('metadata', {})
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.caption(f"📅 Generated: {report_meta.get('generated_at', 'Now')}")
    with col2:
        st.caption(f"🤖 AI Model: {report_meta.get('ai_model', 'Advanced AI')}")
    with col3:
        st.caption(f"📊 Analysis Depth: {report_meta.get('analysis_depth', 'Comprehensive')}")
    
    # Export options
    st.write("### 📥 Export Options")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Download PDF Report", type="secondary"):
            st.info("PDF download feature coming soon!")
    
    with col2:
        if st.button("📧 Email Report", type="secondary"):
            st.info("Email feature coming soon!")
    
    with col3:
        if st.button("🔗 Share Report", type="secondary"):
            st.info("Share feature coming soon!")
    
    # Action buttons for next steps
    st.write("---")
    st.write("### 🎯 Ready for Your Next Learning Adventure?")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔄 Practice More", type="primary", use_container_width=True):
            st.session_state.current_step = 'practice'
            st.rerun()
    
    with col2:
        if st.button("📚 New Topic", type="secondary", use_container_width=True):
            # Reset session for new learning
            st.session_state.material_data = None
            st.session_state.quiz_data = None
            st.session_state.current_step = 'learning'
            st.rerun()
    
    with col3:
        if st.button("🎯 Focused Study", type="secondary", use_container_width=True):
            st.session_state.current_step = 'focused_study'
            st.rerun()
    
    with col4:
        if st.button("🏠 Dashboard", type="secondary", use_container_width=True):
            # Clean reset to dashboard
            for key in list(st.session_state.keys()):
                if key.startswith(('quiz_', 'material_', 'chat_')):
                    del st.session_state[key]
            st.session_state.current_step = 'dashboard'
            st.rerun()


def display_comprehensive_ai_report(report):
    """Display the comprehensive AI-generated report"""
    st.write("## 📋 Comprehensive AI Learning Report")
    
    # Student info
    student_info = report.get('student_info', {})
    st.write(f"**👤 Student:** {student_info.get('name', 'Unknown')}")
    st.write(f"**🎯 Specialization:** {student_info.get('specialization', 'General')}")
    st.write(f"**📊 Current Level:** {student_info.get('current_level', 'Beginner')}")
    st.write(f"**📚 Material Completed:** {student_info.get('material_completed', 'N/A')}")
    
    # Performance summary
    performance = report.get('performance_summary', {})
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Overall Score", f"{performance.get('overall_score', 0)}%")
    with col2:
        st.metric("Performance Level", performance.get('performance_level', 'N/A'))
    with col3:
        status = "PASSED" if performance.get('passed', False) else "NEEDS REVIEW"
        st.metric("Status", status)
    
    # Achievements
    if report.get('achievements'):
        st.write("### 🏆 Achievements Unlocked")
        achievement_cols = st.columns(min(len(report['achievements']), 4))
        for i, achievement in enumerate(report['achievements']):
            col_idx = i % len(achievement_cols)
            with achievement_cols[col_idx]:
                st.write(f"{achievement.get('icon', '🏅')}")
                st.write(f"**{achievement.get('badge', 'Achievement')}**")
                st.caption(achievement.get('description', ''))
    
    # Learning insights
    if report.get('learning_insights'):
        insights = report['learning_insights']
        st.write("### 🧠 AI Learning Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Learning Style:** {insights.get('learning_style', 'balanced').title()}")
            st.info(f"**Strongest Area:** {insights.get('strength_area', 'N/A')}")
        
        with col2:
            st.warning(f"**Focus Area:** {insights.get('improvement_area', 'N/A')}")
            st.info(f"**Progress Trend:** {insights.get('progress_trend', 'steady').title()}")
        
        if insights.get('study_recommendations'):
            st.write("**🎯 Personalized Study Methods:**")
            for rec in insights['study_recommendations']:
                st.write(f"• {rec}")
    
    # AI Recommendations
    if report.get('recommendations'):
        rec = report['recommendations']
        st.write("### 💡 AI-Powered Next Steps")
        
        if rec['status'] == 'ready_to_advance':
            st.success(f"🎉 {rec['message']}")
        elif rec['status'] == 'review_and_advance':
            st.info(f"📚 {rec['message']}")
        else:
            st.warning(f"🔄 {rec['message']}")
        
        if rec.get('recommended_topics'):
            st.write("**📖 Next Topics to Study:**")
            for topic in rec['recommended_topics']:
                st.write(f"• {topic}")
        
        if rec.get('study_approach'):
            st.write("**📋 Recommended Study Approach:**")
            for approach in rec['study_approach']:
                st.write(f"• {approach}")
        
        if rec.get('estimated_timeline'):
            st.info(f"⏱️ **Estimated Timeline:** {rec['estimated_timeline']}")
    
    # Next steps
    if report.get('next_steps'):
        next_steps = report['next_steps']
        st.write("### 🎯 Detailed Action Plan")
        
        st.success(f"**🎯 Immediate Priority:** {next_steps.get('immediate_action', 'Continue learning')}")
        
        if next_steps.get('short_term_goals'):
            st.write("**📅 Short-term Goals (2-4 weeks):**")
            for goal in next_steps['short_term_goals']:
                st.write(f"• {goal}")
        
        if next_steps.get('suggested_resources'):
            st.write("**📚 Recommended Resources:**")
            for resource in next_steps['suggested_resources']:
                st.write(f"• {resource}")
        
        st.info(f"**🌟 Long-term Vision:** {next_steps.get('long_term_vision', 'Continue your learning journey!')}")

# Main app logic
def main():
    # Sidebar with progress
    with st.sidebar:
        st.title("🚀 Learning Progress")
        
        if st.session_state.user_data:
            st.write(f"**Welcome, {st.session_state.user_data['username']}!**")
            
            spec_display = "Data Engineering" if st.session_state.user_data['specialization'] == 'data_engineer' else "Data Science"
            st.write(f"**Specialization:** {spec_display}")
            
            if st.session_state.user_data.get('level'):
                st.write(f"**Level:** {st.session_state.user_data['level'].title()}")
        
        st.write("---")
        
        # Progress steps
        steps = ['login', 'assessment', 'learning', 'quiz', 'results']
        step_names = ['Login', 'Assessment', 'Learning', 'Quiz', 'Results']
        
        current_step_index = steps.index(st.session_state.current_step) if st.session_state.current_step in steps else 0
        
        for i, (step, name) in enumerate(zip(steps, step_names)):
            if i <= current_step_index:
                st.write(f"✅ {name}")
            else:
                st.write(f"⭕ {name}")
        
        st.write("---")
        st.write("Made with ❤️ using Streamlit")
    
    # Main content based on current step
    if st.session_state.current_step == 'login':
        login_page()
    elif st.session_state.current_step == 'assessment':
        assessment_page()
    elif st.session_state.current_step == 'learning':
        learning_page()
    elif st.session_state.current_step == 'quiz':
        quiz_page()
    elif st.session_state.current_step == 'results':
        results_page()

if __name__ == "__main__":
    main()