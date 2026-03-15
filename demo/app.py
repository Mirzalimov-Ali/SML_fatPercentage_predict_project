import pandas as pd
import gradio as gr
import requests

# ===============================
# FASTAPI BACKEND URL
# ===============================
API_URL = "http://127.0.0.1:8000/predict"

# ===============================
# CATEGORY LOGIC
# ===============================
def get_age_category(age):
    if 0 <= age <= 29:
        return "0-29"
    elif 30 <= age <= 39:
        return "30-39"
    elif 40 <= age <= 49:
        return "40-49"
    else:
        return "50-100"

def get_weight_category(weight):
    if 40 <= weight <= 59:
        return "40-59"
    elif 60 <= weight <= 79:
        return "60-79"
    elif 80 <= weight <= 99:
        return "80-99"
    elif 100 <= weight <= 130:
        return "100-130"
    else:
        return "unknown"

def get_bmi_category(bmi):
    if 12.3 <= bmi < 18.5:
        return "under weight"
    elif 18.5 <= bmi <= 24.9:
        return "healthy"
    elif 24.9 < bmi <= 29.9:
        return "overweight"
    elif 29.9 < bmi <= 39.8:
        return "obesity"
    elif 40.1 <= bmi <= 49.8:
        return "sever obesity"
    else:
        return "unknown"

# ===============================
# PREDICT VIA API
# ===============================
def predict_fat_percentage(
    age, gender, weight, height,
    max_bpm, avg_bpm, resting_bpm,
    session_duration, calories_burned,
    workout_type, water_intake,
    workout_frequency, experience_level,
    bmi
):
    age_category = get_age_category(age)
    weight_category = get_weight_category(weight)
    bmi_category = get_bmi_category(bmi)

    payload = {
        "age": age,
        "gender": gender,
        "weight": weight,
        "height": height,
        "max_bpm": max_bpm,
        "avg_bpm": avg_bpm,
        "resting_bpm": resting_bpm,
        "session_duration": session_duration,
        "calories_burned": calories_burned,
        "workout_type": workout_type,
        "water_intake": water_intake,
        "workout_frequency": workout_frequency,
        "experience_level": experience_level,
        "bmi": bmi,
        "age_category": age_category,
        "weight_category": weight_category,
        "bmi_category": bmi_category
    }

    response = requests.post(API_URL, json=payload, timeout=5)
    response.raise_for_status()
    result = response.json()

    return (
        f"{result['fat_percentage']} %",
        result["category"],
        result["meaning"],
        result["recommendation"]
    )

# ===============================
# CUSTOM CSS
# ===============================
CUSTOM_CSS = """
body {
    margin: 0;
}

.gr-box {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(12px);
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.08);
    padding: 30px;
    max-width: 300px;
    width: 100%;
    box-sizing: border-box;
}

h1 {
    text-align: center;
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 20px;
    margin-top: 30px;
}

.gradient-text {
    background: linear-gradient(to right, #38bdf8, #22c55e);
    -webkit-background-clip: text;
    color: transparent;
}

.emoji {
    color: initial !important;
    margin-right: 8px;
}

button {
    background: linear-gradient(135deg, #22c55e, #38bdf8) !important;
    color: black !important;
    font-size: 20px !important;
    font-weight: 700 !important;
    border-radius: 14px !important;
    height: 60px !important;
    width: 100%;
    margin-bottom: 40px;
    margin-top: 10px;
}

label span,
.gr-label span {
    background: transparent !important;
    box-shadow: none !important;
    padding: 0 !important;
}

.gr-row, .gr-column {
    display: flex;
    justify-content: center;
    align-items: center;
    flex-direction: column;
}

footer { 
    display: none !important; 
}
"""

# ===============================
# UI
# ===============================
with gr.Blocks(css=CUSTOM_CSS, theme=gr.themes.Soft()) as demo:

    # ================= Title =================
    gr.Markdown("""
    <h1>
        <span class="emoji">🏋️</span>
        <span class="gradient-text">Fat Percentage Predictor</span>
    </h1>
    """)

    # ================= Inputs =================
    with gr.Row():
        with gr.Column():
            age = gr.Number(label="Age", value=19)
            weight = gr.Number(label="Weight (kg)", value=80)
            max_bpm = gr.Number(label="Max BPM", value=160)
            resting_bpm = gr.Number(label="Resting BPM", value=60)   
            session_duration = gr.Number(label="Session Duration (h)", value=1.00)
            workout_type = gr.Dropdown(["Yoga", "Cardio", "Strength", "HIIT"], value="Strength")
            workout_frequency = gr.Number(label="Workout / week", value=3)
            bmi = gr.Number(label="BMI", value=23.2)

        with gr.Column():
            gender = gr.Dropdown(["Male", "Female"], value="Male")
            height = gr.Number(label="Height (m)", value=1.80)
            avg_bpm = gr.Number(label="Avg BPM", value=160)
            calories_burned = gr.Number(label="Calories Burned", value=1200)
            water_intake = gr.Number(label="Water Intake (L)", value=1.5)
            experience_level = gr.Number(label="Experience Level", value=3)

    # ================= Predict Button =================
    btn = gr.Button("🚀 Predict")

    # ================= Outputs =================
    with gr.Row():
        with gr.Column():
            fat = gr.Textbox(label="Fat Percentage")
            category = gr.Textbox(label="Category")
        with gr.Column():
            meaning = gr.Textbox(label="Meaning")
            recommendation = gr.Textbox(label="Recommendation")

    # ================= Connect button =================
    btn.click(
        predict_fat_percentage,
        inputs=[
            age, gender, weight, height,
            max_bpm, avg_bpm, resting_bpm,
            session_duration, calories_burned,
            workout_type, water_intake,
            workout_frequency, experience_level,
            bmi
        ],
        outputs=[fat, category, meaning, recommendation]
    )

if __name__ == "__main__":
    demo.launch()
