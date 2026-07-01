import streamlit as st
import pdfplumber
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from reportlab.pdfgen import canvas
from io import BytesIO

# ================= UI CONFIG =================

st.set_page_config(
    page_title="AI Resume SaaS",
    layout="wide"
)

st.markdown("""
<style>
.main {
    background-color: #0f172a;
    color: white;
}
.stMetric {
    background-color: #1e293b;
    padding: 10px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

st.title("🚀 AI Resume Intelligence SaaS")
st.markdown("### ATS Scoring • AI Matching • Career Prediction")

# ================= MODEL =================

model = SentenceTransformer('all-MiniLM-L6-v2')

# ================= DATA =================

SKILLS_DB = [
    "python", "java", "c++", "sql", "machine learning",
    "deep learning", "nlp", "pandas", "numpy",
    "tensorflow", "pytorch", "excel", "power bi",
    "communication", "teamwork", "git"
]

JOB_ROLES = {
    "Data Scientist": ["python", "machine learning", "pandas", "numpy"],
    "ML Engineer": ["python", "tensorflow", "pytorch", "deep learning"],
    "Data Analyst": ["sql", "excel", "power bi", "python"],
    "Backend Developer": ["java", "sql", "git"],
}

# ================= FUNCTIONS =================

def extract_text(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
    return text.lower()


def extract_skills(text):
    return [s for s in SKILLS_DB if s in text]


def semantic_score(resume, jd):
    emb = model.encode([resume, jd])
    return round(cosine_similarity([emb[0]], [emb[1]])[0][0] * 100, 2)


def predict_role(skills):
    best_role = "Unknown"
    best_score = 0

    for role, req_skills in JOB_ROLES.items():
        match = len(set(skills) & set(req_skills))
        score = match / len(req_skills)

        if score > best_score:
            best_score = score
            best_role = role

    return best_role, round(best_score * 100, 2)


def hire_probability(score, skills_count):
    return round(min(95, score * 0.7 + skills_count * 3), 2)


def gauge_chart(score):
    fig, ax = plt.subplots()
    ax.pie(
        [score, 100 - score],
        colors=["#22c55e", "#334155"],
        startangle=90,
        wedgeprops={"width": 0.4}
    )
    ax.set_title("ATS Score")
    st.pyplot(fig)


def generate_pdf(score, role, hire_prob, skills):
    buffer = BytesIO()
    c = canvas.Canvas(buffer)

    c.setFont("Helvetica-Bold", 14)
    c.drawString(100, 800, "AI Resume SaaS Report")

    c.setFont("Helvetica", 11)
    c.drawString(100, 770, f"ATS Score: {score}%")
    c.drawString(100, 750, f"Predicted Role: {role}")
    c.drawString(100, 730, f"Hire Probability: {hire_prob}%")

    y = 700
    c.drawString(100, y, "Skills:")
    y -= 20

    for s in skills:
        c.drawString(120, y, f"- {s}")
        y -= 15

    c.save()
    buffer.seek(0)
    return buffer

# ================= SIDEBAR =================

st.sidebar.title("⚙ Controls")
mode = st.sidebar.radio("Mode", ["Single Resume", "Compare Resumes"])

# ================= SINGLE MODE =================

if mode == "Single Resume":

    uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
    job_desc = st.text_area("Paste Job Description")

    if uploaded_file and job_desc:

        resume_text = extract_text(uploaded_file)
        skills = extract_skills(resume_text)

        st.subheader("📄 Resume Preview")
        st.write(resume_text[:1200])

        score = semantic_score(resume_text, job_desc)
        role, role_score = predict_role(skills)
        hire_prob = hire_probability(score, len(skills))

        col1, col2, col3 = st.columns(3)
        col1.metric("ATS Score", f"{score}%")
        col2.metric("Predicted Role", role)
        col3.metric("Hire Probability", f"{hire_prob}%")

        gauge_chart(score)

        st.subheader("📊 Skills Found")
        st.write(skills)

        st.subheader("💡 Insights")
        st.write(f"✔ Best Role Fit: {role}")
        st.write(f"✔ Match Strength: {role_score}%")
        st.write(f"✔ Hire Probability: {hire_prob}%")

        pdf = generate_pdf(score, role, hire_prob, skills)

        st.download_button(
            "📥 Download Report",
            pdf,
            file_name="resume_report.pdf",
            mime="application/pdf"
        )

# ================= COMPARISON MODE =================

else:

    st.subheader("📑 Compare Two Resumes")

    file1 = st.file_uploader("Resume 1", type=["pdf"])
    file2 = st.file_uploader("Resume 2", type=["pdf"])
    job_desc = st.text_area("Job Description")

    if file1 and file2 and job_desc:

        r1 = extract_text(file1)
        r2 = extract_text(file2)

        s1 = semantic_score(r1, job_desc)
        s2 = semantic_score(r2, job_desc)

        col1, col2 = st.columns(2)
        col1.metric("Resume 1 Score", f"{s1}%")
        col2.metric("Resume 2 Score", f"{s2}%")

        winner = "Resume 1" if s1 > s2 else "Resume 2"
        st.success(f"🏆 Better Resume: {winner}")

        fig, ax = plt.subplots()
        ax.bar(["Resume 1", "Resume 2"], [s1, s2])
        st.pyplot(fig)
