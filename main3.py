import streamlit as st
import speech_recognition as sr
import cv2
import os
import time
import tempfile
import threading
from gtts import gTTS
from playsound import playsound
import google.generativeai as genai

# Gemini setup
genai.configure(api_key="AIzaSyCuEYR_kd4Dahj3eXAgPLkgQvcCon7NXDE")
model = genai.GenerativeModel("gemini-pro")
recognizer = sr.Recognizer()

# Capture voice input
def get_voice_input(prompt_text):
    speak_text(prompt_text)
    with sr.Microphone() as source:
        audio = recognizer.listen(source)
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return ""

# Speak text with selected accent
def speak_text(text, accent='us'):
    tts = gTTS(text=text, lang='en', tld=accent)
    tts.save("temp.mp3")
    playsound("temp.mp3")
    os.remove("temp.mp3")

# Generate questions using Gemini
def generate_questions(role, tech, years, q_type="mixed"):
    prompt = f"""
    Generate a {q_type} interview for a {role} with {years} years of experience in {tech}.
    Include at least 5 questions suitable for this profile.
    """
    response = model.generate_content(prompt)
    return [q for q in response.text.split('\n') if q.strip() != '']

# Start webcam and record video
def start_webcam_recording(filename="interview.avi"):
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(filename, fourcc, 20.0, (640, 480))
    return cap, out

# Evaluate answers

def evaluate_answer(question, answer):
    prompt = f"""
    Evaluate the following response:

    Question: {question}
    Answer: {answer}

    Score the user out of 100 based on:
    - Communication Skills
    - Clarity of Thought
    - Technical Knowledge
    - Cultural Fit

    Provide a short feedback for each category and total score.
    """
    response = model.generate_content(prompt)
    return response.text

# Streamlit UI
st.title("AI-Powered Interview Simulator")
st.write("Speak your details to begin the interview.")

if st.button("Start Interview"):
    role = get_voice_input("Which role are you applying for?")
    tech = get_voice_input("What is your primary technology?")
    years = get_voice_input("How many years of experience do you have?")
    q_type = get_voice_input("Do you want a technical, behavioral, or mixed interview?").lower()
    accent = get_voice_input("Choose AI voice accent: US, UK, Indian, Australian")

    if accent.lower() in ['us', 'uk', 'in', 'au']:
        accent_code = accent.lower()
    else:
        accent_code = 'us'

    questions = generate_questions(role, tech, years, q_type)

    st.success("Interview Starting...")

    # Start webcam
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix='.avi')
    cap, out = start_webcam_recording(temp_video.name)

    answers = []

    def capture_video():
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                out.write(frame)
                cv2.imshow('Interview Recording', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

    video_thread = threading.Thread(target=capture_video)
    video_thread.start()

    for i, question in enumerate(questions):
        st.write(f"Q{i+1}: {question}")
        speak_text(question, accent_code)
        start_time = time.time()
        answer = ""

        while time.time() - start_time < 20:
            answer = get_voice_input("Your answer:")
            if answer.strip() != "":
                break

        if answer.strip() == "":
            st.write("No response detected. Moving to next question.")
            answers.append((question, "No Answer"))
        else:
            answers.append((question, answer))

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    st.success("Interview Completed! Generating Feedback...")

    for question, answer in answers:
        if answer != "No Answer":
            feedback = evaluate_answer(question, answer)
            st.write(f"**Question**: {question}")
            st.write(f"**Your Answer**: {answer}")
            st.write(f"**Feedback**: {feedback}")
        else:
            st.write(f"**Question**: {question}")
            st.write("**Your Answer**: No response")
            st.write("**Feedback**: Skipped")

    st.video(temp_video.name)
    st.success("Interview video saved and feedback generated.")
