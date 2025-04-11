# 🎙️AI-interview-simulator---Prototype


**AI-powered mock interview platform** designed to help candidates practice and improve their interview skills. This simulator provides real-time voice-based Q&A, evaluates responses using Gemini API, and gives personalized feedback across multiple metrics like communication, technical knowledge, clarity, and more.

---

## 🚀 Features

- 🎤 **Voice-based interaction** (Speech-to-Text + Text-to-Speech)
- 👁️ **Automatic camera activation** with OpenCV
- ⏱️ **Timed responses** (20 seconds per answer)
- 🤖 **AI-powered evaluation** using Gemini API
- 📊 **Feedback metrics**:
  - Communication Skills
  - Clarity of Thought
  - Technical Knowledge
  - Problem-solving Ability
- 🌍 **Customizable voice ethnicity** for diverse AI voices
- 🎯 Real-time scoring and summary at the end

---

## 🧠 Tech Stack

- **Frontend**: Streamlit / Gradio (optional)
- **Backend**: Python
- **AI Evaluation**: Gemini API (Google)
- **Speech Recognition**: `speech_recognition` + Google STT
- **Voice Output**: `pyttsx3` or Google TTS
- **Video Processing**: OpenCV
- **Camera Integration**: Auto-detection and streaming

---

## 📦 Installation

```bash
git clone https://github.com/yourusername/ai-interview-simulator.git
cd ai-interview-simulator
pip install -r requirements.txt
