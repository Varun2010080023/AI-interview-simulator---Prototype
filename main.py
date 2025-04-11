import speech_recognition as sr
import pyttsx3
import cv2
import threading
import queue
import time
import json
import os
import numpy as np
from google.generativeai import GenerativeModel, configure
import google.generativeai as genai
from dotenv import load_dotenv

class InterviewSimulator:
    def __init__(self):
        # Loading environment variables for API keys
        load_dotenv()
        
        # Configure Gemini API
        api_key = os.getenv("api_key", "")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        configure(api_key=api_key)
        
        # Initialize Gemini model
        self.gemini_model = GenerativeModel('gemini-1.5-pro')
        
        # Initialize the speech recognition and text-to-speech engines
        self.recognizer = sr.Recognizer()
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 175)  # Speed of speech
        
        # Set up voices (you can modify these settings based on available voices)
        voices = self.engine.getProperty('voices')
        if len(voices) > 1:
            self.engine.setProperty('voice', voices[1].id)  # Use a different voice if available
        
        # Interview state variables
        self.current_interview = None
        self.current_question_index = 0
        self.interview_responses = []
        self.interview_results = {}
        self.interview_in_progress = False
        
        # Video capture
        self.video_enabled = False
        self.cap = None
        
        # Audio processing queue
        self.audio_queue = queue.Queue()
        
        # Ensure data directory exists
        if not os.path.exists('data'):
            os.makedirs('data')
    
    def start_video_capture(self):
        """Initialize and start the video capture"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open camera")
            return False
        
        self.video_enabled = True
        threading.Thread(target=self._video_thread, daemon=True).start()
        return True
    
    def _video_thread(self):
        """Thread for capturing and displaying video"""
        while self.video_enabled and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # Add interview UI elements to the frame
            if self.interview_in_progress:
                # Add text for current question number
                question_text = f"Question {self.current_question_index + 1}"
                cv2.putText(frame, question_text, (20, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Add recording indicator when listening
                if hasattr(self, 'is_listening') and self.is_listening:
                    cv2.circle(frame, (frame.shape[1] - 30, 30), 10, (0, 0, 255), -1)
            
            # Display the frame
            cv2.imshow('Interview Simulation', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Clean up
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.video_enabled = False
    
    def stop_video_capture(self):
        """Stop video capture"""
        self.video_enabled = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
    
    def speak_text(self, text):
        """Convert text to speech and speak it"""
        print(f"AI: {text}")
        self.engine.say(text)
        self.engine.runAndWait()
    
    def listen_for_response(self, timeout=30):
        """Listen for user's voice response with a timeout"""
        self.is_listening = True
        print("Listening...")
        
        try:
            with sr.Microphone() as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                audio = self.recognizer.listen(source, timeout=timeout)
                
            print("Processing speech...")
            response = self.recognizer.recognize_google(audio)
            print(f"You said: {response}")
            return response
        except sr.WaitTimeoutError:
            print("Timeout - no speech detected")
            return ""
        except sr.UnknownValueError:
            print("Could not understand audio")
            return ""
        except sr.RequestError as e:
            print(f"Error with speech recognition service: {e}")
            return ""
        finally:
            self.is_listening = False
    
    async def generate_interview_questions(self, technology, role, experience_level, interview_type):
        """Generate interview questions using Gemini AI"""
        prompt = f"""
        Generate an interview for a {experience_level} level {role} position focused on {technology}.
        This should be a {interview_type} interview (technical, behavioral, or mixed).
        
        For each question, provide:
        1. The question text
        2. A model answer or key points that should be covered
        3. Keywords to listen for in a good answer
        
        Format the response as a JSON array of question objects with the following structure:
        [
            {{
                "question": "Question text here",
                "expected_answer": "Model answer or key points here",
                "keywords": ["keyword1", "keyword2", "keyword3"]
            }},
            ...
        ]
        
        Include 5 questions. Make the questions appropriately challenging for a {experience_level} level candidate.
        """
        
        try:
            response = await self.gemini_model.generate_content_async(prompt)
            text_response = response.text
            
            # Extract JSON from the response
            # Find JSON array pattern in the response
            import re
            json_match = re.search(r'\[\s*{.*}\s*\]', text_response, re.DOTALL)
            
            if json_match:
                questions_json = json_match.group(0)
                questions = json.loads(questions_json)
            else:
                # Fallback - try to parse the whole response
                questions = json.loads(text_response)
            
            return questions
        except Exception as e:
            print(f"Error generating questions: {e}")
            # Return some default questions as fallback
            return [
                {
                    "question": f"Tell me about your experience with {technology}.",
                    "expected_answer": "The candidate should describe their relevant experience.",
                    "keywords": ["experience", technology.lower(), "project", "skills"]
                },
                {
                    "question": "What are your strengths and weaknesses?",
                    "expected_answer": "The candidate should provide honest self-assessment.",
                    "keywords": ["strength", "weakness", "improve", "learn", "challenge"]
                }
            ]
    
    async def evaluate_response(self, question, user_response, expected_answer, keywords):
        """Use Gemini to evaluate the user's response"""
        if not user_response:
            return {
                "score": 0,
                "feedback": "No response provided."
            }
        
        prompt = f"""
        Evaluate the following interview response:
        
        Question: {question}
        
        Expected answer should cover: {expected_answer}
        
        Key concepts to listen for: {', '.join(keywords)}
        
        Candidate's actual response: {user_response}
        
        Please:
        1. Score the response from 0-100
        2. Provide specific feedback on strengths
        3. Provide specific areas for improvement
        4. Return the result as a JSON object with fields: score, feedback
        """
        
        try:
            response = await self.gemini_model.generate_content_async(prompt)
            text_response = response.text
            
            # Extract JSON from the response
            import re
            json_match = re.search(r'{.*}', text_response, re.DOTALL)
            
            if json_match:
                eval_json = json_match.group(0)
                evaluation = json.loads(eval_json)
            else:
                # Try to parse the whole response
                evaluation = json.loads(text_response)
                
            return evaluation
        except Exception as e:
            print(f"Error evaluating response: {e}")
            # Return a default evaluation
            return {
                "score": 50,
                "feedback": "Unable to provide detailed evaluation due to an error."
            }
    
    async def conduct_interview(self, technology, role, experience_level, interview_type):
        """Conduct a full interview with the user"""
        # Start video if not already started
        if not self.video_enabled:
            if not self.start_video_capture():
                self.speak_text("Unable to access camera. Continuing with audio only.")
        
        self.interview_in_progress = True
        self.speak_text(f"Welcome to your {interview_type} interview for the {experience_level} {role} position with focus on {technology}.")
        time.sleep(1)
        self.speak_text("I'll ask you a series of questions. Please answer each question clearly. Let's begin.")
        time.sleep(1)
        
        # Generate questions
        self.speak_text("Generating questions based on your requirements...")
        questions = await self.generate_interview_questions(technology, role, experience_level, interview_type)
        
        # Initialize interview data
        self.current_interview = {
            "technology": technology,
            "role": role,
            "experience_level": experience_level,
            "interview_type": interview_type,
            "questions": questions
        }
        self.current_question_index = 0
        self.interview_responses = []
        
        # Ask each question and record responses
        total_score = 0
        for idx, question_data in enumerate(questions):
            self.current_question_index = idx
            question = question_data["question"]
            
            # Ask the question
            self.speak_text(f"Question {idx + 1}: {question}")
            
            # Listen for response
            user_response = self.listen_for_response(timeout=120)
            
            # Evaluate response
            self.speak_text("Thank you. Let me evaluate your response.")
            evaluation = await self.evaluate_response(
                question, 
                user_response, 
                question_data["expected_answer"], 
                question_data["keywords"]
            )
            
            # Store response and evaluation
            self.interview_responses.append({
                "question": question,
                "user_response": user_response,
                "evaluation": evaluation
            })
            
            total_score += evaluation["score"]
            
            # Optional: Provide immediate feedback
            if idx < len(questions) - 1:  # Don't provide feedback after the last question
                self.speak_text(f"Moving on to the next question.")
        
        # Calculate final score
        avg_score = total_score / len(questions)
        
        # Provide final feedback
        self.speak_text("Thank you for completing the interview.")
        self.speak_text(f"Your overall score is {avg_score:.1f} out of 100.")
        
        # Generate detailed feedback
        await self.provide_detailed_feedback()
        
        # End interview
        self.interview_in_progress = False
        self.speak_text("The interview simulation has ended. Thank you for participating.")
        
        # Stop video if it was started
        self.stop_video_capture()
        
        return {
            "score": avg_score,
            "responses": self.interview_responses
        }
    
    async def provide_detailed_feedback(self):
        """Provide detailed feedback on the entire interview"""
        if not self.interview_responses:
            self.speak_text("No interview data available for feedback.")
            return
        
        # Prepare data for feedback
        interview_summary = {
            "position": f"{self.current_interview['experience_level']} {self.current_interview['role']}",
            "focus": self.current_interview['technology'],
            "interview_type": self.current_interview['interview_type'],
            "responses": [
                {
                    "question": r["question"],
                    "response": r["user_response"],
                    "score": r["evaluation"]["score"],
                    "feedback": r["evaluation"]["feedback"]
                }
                for r in self.interview_responses
            ]
        }
        
        # Generate comprehensive feedback
        prompt = f"""
        Provide detailed feedback on this completed interview:
        
        {json.dumps(interview_summary, indent=2)}
        
        Include:
        1. Overall performance assessment
        2. Top 3 strengths demonstrated
        3. Top 3 areas for improvement
        4. Specific advice for improving answers to questions with low scores
        5. Next steps for preparation
        
        Make the feedback constructive, specific, and actionable.
        """
        
        try:
            response = await self.gemini_model.generate_content_async(prompt)
            detailed_feedback = response.text
            
            # Speak the feedback in segments for better listening experience
            paragraphs = detailed_feedback.split('\n\n')
            for paragraph in paragraphs:
                if paragraph.strip():
                    self.speak_text(paragraph)
                    time.sleep(0.5)
                    
            return detailed_feedback
        except Exception as e:
            print(f"Error generating detailed feedback: {e}")
            self.speak_text("I experienced some difficulties generating detailed feedback. Here's a simple summary:")
            
            # Calculate average score
            avg_score = sum(r["evaluation"]["score"] for r in self.interview_responses) / len(self.interview_responses)
            
            if avg_score >= 80:
                self.speak_text("You performed very well in this interview. Your responses were thorough and demonstrated good knowledge.")
            elif avg_score >= 60:
                self.speak_text("You did well but there's room for improvement. Consider preparing more specific examples and technical details.")
            else:
                self.speak_text("You should focus on further preparation. Try to be more specific in your answers and review the core concepts.")
                
            return "Basic feedback provided due to error."

    async def save_interview_results(self, filename=None):
        """Save interview results to a file"""
        if not self.interview_responses:
            print("No interview data to save")
            return False
            
        if filename is None:
            # Generate filename based on interview details
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            role = self.current_interview["role"].replace(" ", "_")
            level = self.current_interview["experience_level"]
            filename = f"interview_{role}_{level}_{timestamp}.json"
        
        # Calculate overall score
        total_score = sum(r["evaluation"]["score"] for r in self.interview_responses)
        avg_score = total_score / len(self.interview_responses)
        
        # Prepare data to save
        data = {
            "interview_details": {
                "technology": self.current_interview["technology"],
                "role": self.current_interview["role"],
                "experience_level": self.current_interview["experience_level"],
                "interview_type": self.current_interview["interview_type"],
                "date": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "overall_score": avg_score,
            "questions_and_responses": self.interview_responses
        }
        
        try:
            # Ensure directory exists
            os.makedirs("interview_results", exist_ok=True)
            filepath = os.path.join("interview_results", filename)
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
                
            print(f"Interview results saved to {filepath}")
            return True
        except Exception as e:
            print(f"Error saving interview results: {e}")
            return False

async def main():
    interviewer = InterviewSimulator()
    
    # Welcome message
    interviewer.speak_text("Welcome to the AI Interview Simulator.")
    interviewer.speak_text("I'll help you prepare for technical and behavioral interviews.")
    
    # Get interview parameters through voice
    interviewer.speak_text("What technology would you like to focus on? For example, Python, JavaScript, or Data Science.")
    technology = interviewer.listen_for_response()
    if not technology:
        technology = "general programming"
        interviewer.speak_text(f"I'll focus on {technology}.")
    
    interviewer.speak_text("What role are you applying for? For example, Software Engineer, Data Scientist, or Product Manager.")
    role = interviewer.listen_for_response()
    if not role:
        role = "software developer"
        interviewer.speak_text(f"I'll simulate an interview for a {role} position.")
    
    interviewer.speak_text("What experience level? Junior, mid-level, or senior?")
    experience = interviewer.listen_for_response().lower()
    if not experience or experience not in ["junior", "mid", "mid-level", "senior"]:
        experience = "mid-level"
        interviewer.speak_text(f"I'll prepare a {experience} interview.")
    elif experience == "mid":
        experience = "mid-level"
    
    interviewer.speak_text("What type of interview would you like? Technical, behavioral, or mixed?")
    interview_type = interviewer.listen_for_response().lower()
    if not interview_type or interview_type not in ["technical", "behavioral", "mixed"]:
        interview_type = "mixed"
        interviewer.speak_text(f"I'll conduct a {interview_type} interview.")
    
    # Confirm details
    interviewer.speak_text(f"I'll now conduct a {interview_type} interview for a {experience} {role} position with focus on {technology}.")
    interviewer.speak_text("The interview will begin shortly. I'll ask you questions and evaluate your responses.")
    time.sleep(2)
    
    # Conduct the interview
    results = await interviewer.conduct_interview(technology, role, experience, interview_type)
    
    # Save results
    await interviewer.save_interview_results()
    
    interviewer.speak_text("Thank you for using the AI Interview Simulator. Good luck with your job search!")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
