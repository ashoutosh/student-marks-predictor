import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# -------------------- Page configuration --------------------
st.set_page_config(
    page_title="Student Marks Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------- Load trained model safely --------------------
MODEL_PATH = Path(__file__).parent / "student_marks_model.pkl"

@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)

if not MODEL_PATH.exists():
    st.error(
        "The trained model file 'student_marks_model.pkl' was not found. "
        "Place it in the same folder as app.py."
    )
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
except Exception as error:
    st.error(f"The trained model could not be loaded: {error}")
    st.stop()

# -------------------- Session state --------------------
if "history" not in st.session_state:
    st.session_state.history = []

if "latest_prediction" not in st.session_state:
    st.session_state.latest_prediction = None

# -------------------- Sidebar navigation --------------------
with st.sidebar:
    st.title("🎓 Student Predictor")
    st.write("Your academic performance dashboard.")
    st.divider()

    page = st.radio(
        "🧭 Choose a section:",
        ["🏠 Dashboard", "📊 Prediction History", "ℹ️ About Project"],
    )

    st.divider()
    st.subheader("🧮 How It Works")
    st.write(
        "Enter your daily study hours and attendance. "
        "The trained regression model estimates a mark."
    )
    st.info(
        "This is an educational demonstration trained on a small, "
        "illustrative dataset. Predictions are not validated for real-world use."
    )
    st.divider()
    st.caption("Built with Python, Pandas, scikit-learn and Streamlit 🚀")

# -------------------- Dashboard --------------------
if page == "🏠 Dashboard":
    st.title("🎓 Student Marks Predictor")
    st.markdown(
        """
        ### Welcome to your academic dashboard! 📚
        Enter your study hours and attendance to generate an estimated mark.
        ---
        """
    )

    left, right = st.columns(2)
    with left:
        hours = st.number_input(
            "Daily study hours",
            min_value=0.0,
            max_value=24.0,
            value=2.0,
            step=0.5,
        )
    with right:
        attendance = st.slider(
            "Attendance percentage",
            min_value=0,
            max_value=100,
            value=75,
        )

    if st.button("🔮 Predict My Marks", type="primary"):
        input_data = pd.DataFrame(
            {
                "study_hours": [hours],
                "attendance": [attendance],
            }
        )

        try:
            predicted = float(model.predict(input_data)[0])
            marks = max(0.0, min(100.0, predicted))

            record = {
                "Study Hours": hours,
                "Attendance (%)": attendance,
                "Estimated Marks": round(marks, 1),
            }
            st.session_state.history.append(record)
            st.session_state.latest_prediction = record
        except Exception as error:
            st.error(f"Could not generate a prediction: {error}")

    result = st.session_state.latest_prediction

    if result is None:
        st.info("Choose your study hours and attendance, then click **Predict My Marks**.")
    else:
        marks = float(result["Estimated Marks"])
        st.success(f"Your estimated marks: {marks:.1f}/100")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🎯 Estimated Marks", f"{marks:.1f}/100")
        with col2:
            st.metric("📚 Daily Study Hours", f'{result["Study Hours"]:.1f} hours')
        with col3:
            st.metric("📅 Attendance", f'{result["Attendance (%)"]}%')

        st.progress(int(marks))
        st.caption(
            "Educational demonstration only. The model has not been validated "
            "for accurate real-world student mark prediction."
        )

        st.subheader("📚 Study Recommendations")
        if marks >= 80:
            st.write("🌟 **Excellent estimated result.**")
            st.write("- Practise advanced questions.")
            st.write("- Revise important concepts regularly.")
            st.write("- Take timed mock tests.")
        elif marks >= 60:
            st.write("👍 **A good starting point.**")
            st.write("- Identify and revise weak topics.")
            st.write("- Practise challenging questions.")
            st.write("- Review mistakes from previous tests.")
        elif marks >= 40:
            st.write("📖 **Focus on strengthening your fundamentals.**")
            st.write("- Revise basic concepts first.")
            st.write("- Practise questions every day.")
            st.write("- Review mistakes and retry questions.")
        else:
            st.write("💪 **Build a steady study routine.**")
            st.write("- Start with the basics of each topic.")
            st.write("- Study in short, consistent sessions.")
            st.write("- Solve easier questions before harder ones.")

        st.subheader("📈 Estimated Marks at Different Study Hours")
        chart_hours = list(range(0, 25, 2))
        chart_input = pd.DataFrame(
            {
                "study_hours": chart_hours,
                "attendance": [int(result["Attendance (%)"])] * len(chart_hours),
            }
        )
        chart_marks = model.predict(chart_input)
        chart_data = pd.DataFrame(
            {
                "Study Hours": chart_hours,
                "Estimated Marks": [
                    max(0.0, min(100.0, float(value))) for value in chart_marks
                ],
            }
        )
        st.line_chart(chart_data, x="Study Hours", y="Estimated Marks")
        st.caption(
            f"Chart uses {result['Attendance (%)']}% attendance and the trained model "
            "to estimate marks for different study-hour values."
        )

# -------------------- Prediction History --------------------
elif page == "📊 Prediction History":
    st.title("📊 Prediction History")
    st.write("View or download predictions made during this session.")

    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, use_container_width=True, hide_index=True)

        csv_data = history_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Prediction History",
            data=csv_data,
            file_name="prediction_history.csv",
            mime="text/csv",
        )

        if st.button("🗑️ Clear Prediction History"):
            st.session_state.history = []
            st.session_state.latest_prediction = None
            st.rerun()
    else:
        st.info("No predictions yet. Go to Dashboard to make your first prediction.")

# -------------------- About Project --------------------
elif page == "ℹ️ About Project":
    st.title("ℹ️ About Student Marks Predictor")
    st.write(
        "This beginner project uses a trained Linear Regression model to estimate "
        "marks from two inputs: daily study hours and attendance percentage."
    )

    st.subheader("🛠️ Technologies Used")
    st.write("- Python")
    st.write("- Streamlit")
    st.write("- Pandas")
    st.write("- scikit-learn")
    st.write("- Joblib (to save and load the trained model)")

    st.subheader("⚙️ How it works")
    st.write(
        "The model was trained in train_model.py. The app loads the saved "
        "student_marks_model.pkl file and passes study hours and attendance "
        "to the model when you click Predict My Marks."
    )

    st.warning(
        "The training data is small and illustrative. Evaluation results from it "
        "do not establish real-world accuracy. Use a larger, suitable dataset "
        "and evaluate it carefully before making meaningful predictions."
    )
