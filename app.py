import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import tempfile

# ----------------- PAGE CONFIG -----------------
st.set_page_config(page_title="CV Master", layout="centered", page_icon="📄")

# ----------------- SIDEBAR -----------------
with st.sidebar:
    col1, col2, col3 = st.columns([2, 3, 2])  # Increase 3rd column width
    with col2:
        st.image("logo.png", width=100)

    st.markdown(
        """
        <p style='text-align: justify;'>
        <strong>CV Master</strong> is an AI-powered CV analysis platform that helps students and job seekers
        evaluate resumes, identify skill gaps, get grammar feedback and prepare for future applications.
        </p>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown(
        """
        <p style='text-align: center;'>
        Tanmoy Giri, M.Tech, CSDP<br>
        Batch of 2024-26, IIT Kharagpur
        </p>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.markdown(
        "[LinkedIn](https://www.linkedin.com/in/iamtgiri) | "
        "[Source Code](https://github.com/yourusername/yourrepo) | "
        "[Portfolio](https://iamtgiri.github.io)",
        unsafe_allow_html=True
    )
    st.info("You can clone the repository from GitHub and run it locally.")

# ----------------- HEADER -----------------
col1, col2, col3 = st.columns([20, 10, 20])  # Increase 3rd column width
with col2:
    st.image("logo.png", width=150)

st.markdown("<h4 style='text-align: center;'>Evaluation, Skill Analysis, Scoring & Grammar Review</h4>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# ----------------- API KEY -----------------
api_key = st.text_input(
    "Enter your Gemini API Key and press ENTER",
    type="password",
    value=os.getenv("GOOGLE_API_KEY", "")
)

with st.expander("ℹ️ How to get a Gemini API Key"):
    st.markdown(
        """
        1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey).
        2. Sign in with your Google account.
        3. Click **Get API Key** → **+ Create API Key**.
        4. Copy and paste the key above.
        """
    )

if not api_key:
    st.warning("Please enter your Google API key to proceed.")
    st.stop()

# ----------------- MODEL -----------------
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.7,
    google_api_key=api_key
)
parser = StrOutputParser()

# ----------------- PROMPTS -----------------
cv_eval_prompt = PromptTemplate(
    template="""You are an experienced interviewer and hiring manager recruiting for the role of **{role}**. 
Given the following CV, perform a detailed evaluation covering the following aspects:

1. **CV Assessment** - Analyze the CV with respect to the {role} position. Evaluate the candidate's relevant skills, experience level, strengths, weaknesses, and overall impression.
2. **Suggestions for Improvement** - Recommend specific, actionable changes to improve the CV's relevance, clarity, and impact for a {role} role. Include keyword optimization suggestions.
3. **Customized Interview Questions** - Generate a list of insightful, role-specific interview questions tailored to the CV content.
4. **Practice Advice for Candidate** - Suggest technical and non-technical topics or questions the candidate should prepare for, based on observed gaps or emphasis in the CV.

Here is the CV:
{cv}
""",
    input_variables=["cv", "role"]
)

skill_extraction_prompt = PromptTemplate(
    template="""Extract all relevant technical and soft skills from the following text. 
Return them as a **comma-separated list** without extra commentary.

Text:
{content}
""",
    input_variables=["content"]
)

skill_gap_prompt = PromptTemplate(
    template="""
    Compare the following skills:

    **CV Skills:** {cv_skills}
    **Job Description Skills:** {jd_skills}

    Identify:
    1. Skills in JD missing from CV.
    2. Skills in CV not required in JD.
    3. Transferable/related skills in CV that match JD requirements.

    Return in **Markdown** format.
    """,
    input_variables=["cv_skills", "jd_skills"]
)

scoring_prompt = PromptTemplate(
    template="""You are a professional recruiter. Score the following CV for the role of **{role}** based on these parameters:

1. **Relevance to Role**
2. **Clarity**
3. **Skill Match**
4. **Readability**

For each parameter, give:
- A score out of 10
- A star rating out of 5 (e.g., ⭐⭐⭐⭐☆)

Finally, provide:
- An **Overall Score** out of 10
- A short summary sentence about the candidate's fit.

CV Content:
{cv}
""",
    input_variables=["cv", "role"]
)

grammar_prompt = PromptTemplate(
    template="""You are an expert proofreader and career coach. 
Review the following CV text for **grammar mistakes, awkward phrasing, formatting issues, and layout inconsistencies**. 
Provide feedback in these sections:

1. **Grammar & Spelling Errors**
2. **Sentence Clarity**
3. **Formatting Suggestions**

CV Content:
{cv}
""",
    input_variables=["cv"]
)

# ----------------- CHAINS -----------------
cv_eval_chain = cv_eval_prompt | model | parser
skill_extraction_chain = skill_extraction_prompt | model | parser
skill_gap_chain = skill_gap_prompt | model | parser
scoring_chain = scoring_prompt | model | parser
grammar_chain = grammar_prompt | model | parser

# ----------------- USER INPUTS -----------------
job_role = st.selectbox(
    "Select the job role you are preparing this CV for",
    ["Software Engineer", "Data Scientist", "AI Engineer", "Other"],
    index=0
)

if job_role == "Other":
    job_role = st.text_input(
        "Specify Job Role",
        placeholder="Enter your target role e.g. Backend Developer"
    )

if not job_role.strip():
    st.warning("Please enter a job role to proceed.")
    st.stop()

jd_text = ""
if st.checkbox("Add Job Description for Skill Gap Analysis"):
    jd_text = st.text_area("Paste Job Description text here (optional)", height=200)

uploaded_cv = st.file_uploader("Upload your CV (PDF)", type=["pdf"])

# ----------------- PROCESSING -----------------
if uploaded_cv and job_role:
    with st.spinner("Processing CV..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_cv.read())
            tmp_file_path = tmp_file.name

        try:
            loader = PyPDFLoader(tmp_file_path)
            documents = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            merged_cv = "\n".join(
                chunk.page_content for chunk in splitter.split_documents(documents)
            )

            # Run evaluations
            result = cv_eval_chain.invoke({"cv": merged_cv, "role": job_role})
            scoring_result = scoring_chain.invoke({"cv": merged_cv, "role": job_role})
            grammar_result = grammar_chain.invoke({"cv": merged_cv})

            st.success("✅ CV evaluation complete!")

            st.markdown("### 📊 CV Scoring & Rating")
            st.markdown(scoring_result)

            st.markdown("### 📋 Evaluation Summary")
            st.markdown(result)

            st.markdown("### ✏️ Grammar & Formatting Feedback")
            st.markdown(grammar_result)

            if jd_text.strip():
                st.info("📊 Running skill extraction and gap analysis...")
                cv_skills = skill_extraction_chain.invoke({"content": merged_cv})
                jd_skills = skill_extraction_chain.invoke({"content": jd_text})
                skill_gap_result = skill_gap_chain.invoke(
                    {"cv_skills": cv_skills, "jd_skills": jd_skills}
                )
                st.markdown("### 🛠 Skill Gap Analysis")
                st.markdown(skill_gap_result)

            # Downloads
            st.download_button(
                "📥 Download CV Evaluation",
                result,
                "CV_Evaluation.md",
                "text/markdown"
            )
            st.download_button(
                "📥 Download Scoring Report",
                scoring_result,
                "CV_Scoring.md",
                "text/markdown"
            )
            st.download_button(
                "📥 Download Grammar & Formatting Report",
                grammar_result,
                "Grammar_Formatting_Check.md",
                "text/markdown"
            )
            if jd_text.strip():
                st.download_button(
                    "📥 Download Skill Gap Report",
                    skill_gap_result,
                    "Skill_Gap_Analysis.md",
                    "text/markdown"
                )

        finally:
            os.remove(tmp_file_path)
