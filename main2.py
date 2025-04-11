import os
import cv2
import openai
import speech_recognition as sr
import pyttsx3
import time
import datetime

# Initialize Gemini API
import google.generativeai as genai
genai.configure(api_key="")
model = genai.GenerativeModel("gemini-pro")

# Initialize TTS and STT
recognizer = sr.Recognizer()
tts = pyttsx3.init()

# Video settings
output_video_path = "interview_recording.avi"
frame_width = 640
frame_height = 480
fps = 20

# Function: Speak text
def speak(text):
    print("Bot:", text)
    tts.say(text)
    tts.runAndWait()

# Function: Accurate speech recognition
def listen(retry_attempts=3):
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print("Listening... Please speak clearly.")
        for attempt in range(retry_attempts):
            try:
                audio = recognizer.listen(source, timeout=10, phrase_time_limit=15)
                response = recognizer.recognize_google(audio)
                print("You said:", response)
                return response
            except sr.UnknownValueError:
                print("Sorry, I couldn't understand you.")
                speak("Sorry, I couldn't understand you. Please try again.")
            except sr.WaitTimeoutError:
                print("Listening timed out.")
                speak("I didn't hear anything. Please try again.")
            except sr.RequestError:
                print("Speech service is currently unavailable.")
                speak("Speech service is currently unavailable. Please check your internet.")
                break
    return ""

# Function: Get number of questions
def get_number_of_questions():
    speak("How many interview questions would you like to be asked?")
    while True:
        response = listen()
        if response:
            for word in response.split():
                if word.isdigit():
                    return int(word)
            speak("Please say a number.")
        else:
            speak("Let's try again. How many questions?")

# Function: Ask interview questions
def conduct_interview(num_questions):
    cap = cv2.VideoCapture(0)
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))
    
    speak("Welcome to your AI Interview. Let's begin.")

    for i in range(num_questions):
        # Record video frame
        ret, frame = cap.read()
        if ret:
            out.write(frame)
            cv2.imshow("Interview", frame)

        # Generate a question
        prompt = f"Generate a professional interview question {i+1} for a software engineer."
        response = model.generate_content(prompt)
        question = response.text.strip()
        speak(f"Question {i+1}: {question}")

        # Listen to user's answer
        user_answer = listen()

        # Save another frame during user's answer
        ret, frame = cap.read()
        if ret:
            out.write(frame)

        # Provide simple feedback
        feedback_prompt = f"This was the interview question: '{question}' and the candidate answered: '{user_answer}'. Provide short feedback on the answer."
        feedback = model.generate_content(feedback_prompt)
        speak("Here's some feedback on your answer.")
        speak(feedback.text.strip())

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    speak("Interview complete. Video has been saved.")

# MAIN
if __name__ == "__main__":
    num_qs = get_number_of_questions()
    conduct_interview(num_qs)
    speak("Thank you for participating in the interview. Goodbye!")
