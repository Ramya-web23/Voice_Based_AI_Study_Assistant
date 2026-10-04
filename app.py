
import os
import tempfile
import gradio as gr
import speech_recognition as sr

from google import genai
from gtts import gTTS


# Gemini API
API_KEY = os.getenv("GEMINI_API_KEY")

client = genai.Client(
    api_key=API_KEY
)


# Voice Assistant Prompt
SYSTEM_PROMPT = """
You are a friendly and intelligent voice assistant.

Your job is to have natural and helpful conversations with the user.

Guidelines:
- Understand the user's question carefully.
- Give clear and accurate answers.
- Keep responses concise because they will be converted into speech.
- Use simple and natural language.
- Avoid unnecessary headings and bullet points.
- Do not give extremely long answers unless the user asks for details.
- Explain difficult concepts step-by-step with simple examples.
- Be friendly, conversational and confident.
- If the question is unclear, ask a short clarification question.
- Do not repeatedly say that you are an AI.
"""


# Speech to Text
def speech_to_text(audio):

    recognizer = sr.Recognizer()

    try:

        with sr.AudioFile(audio) as source:
            audio_data = recognizer.record(source)

        text = recognizer.recognize_google(audio_data)

        return text

    except sr.UnknownValueError:

        return "Sorry, I couldn't understand your voice."

    except Exception as e:

        return f"Error: {e}"


# Text to Speech
def text_to_speech(text):

    output_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp3"
    ).name

    tts = gTTS(
        text=text,
        lang="en"
    )

    tts.save(output_file)

    return output_file


# Voice Chatbot
def voice_chat(audio):

    # STT
    user_text = speech_to_text(audio)

    if user_text.startswith("Sorry") or user_text.startswith("Error"):
        return user_text, None

    # Gemini
    response = client.models.generate_content(

        model="gemini-3.6-flash",

        contents=SYSTEM_PROMPT +
        "\n\nUser: " + user_text
    )

    bot_response = response.text

    # TTS
    audio_response = text_to_speech(
        bot_response
    )

    return bot_response, audio_response


# Gradio Interface
with gr.Blocks() as app:

    gr.Markdown(
        "# 🎙️ Voice AI Assistant"
    )

    gr.Markdown(
        "Speak into the microphone and get an AI voice response."
    )

    audio_input = gr.Audio(
        sources=["microphone"],
        type="filepath",
        label="🎤 Speak"
    )

    send_button = gr.Button(
        "Send"
    )

    text_output = gr.Textbox(
        label="🤖 Assistant Response"
    )

    audio_output = gr.Audio(
        label="🔊 Voice Response",
        autoplay=True
    )

    send_button.click(

        fn=voice_chat,

        inputs=audio_input,

        outputs=[
            text_output,
            audio_output
        ]
    )


# Render configuration
if __name__ == "__main__":

    port = int(
        os.getenv("PORT", 7860)
    )

    app.launch(
        server_name="0.0.0.0",
        server_port=port
    )
