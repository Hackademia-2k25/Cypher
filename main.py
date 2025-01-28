import os
from dotenv import load_dotenv
import time
from simple_transcriber import SimpleTranscriber
import sys
sys.path.append('.venv/rags')  # Add the path to where retriever_conversational.ipynb is located

# Import the functions from retriever_conversational
from retriever_conversational import def_explain

def handle_transcription(transcription):
    """Callback function to handle new transcriptions and interact with RAG system."""
    print(f"User query: {transcription}")
    
    # Get response from the RAG system using def_explain from retriever_conversational
    response = def_explain(transcription, "default")
    
    if response:
        print(f"AI response: {response}")
        return response
    else:
        print("No response from AI.")
        return "I'm sorry, I couldn't generate a response."

def main():
    # Initialize the transcriber with a callback to handle transcriptions
    transcriber = SimpleTranscriber(callback=handle_transcription)

    try:
        # Start the recording and transcription process
        transcriber.start_recording()
        print("Recording started. Speak to ask questions. Press Ctrl+C to stop.")
        
        while True:
            # Keep the main thread running
            time.sleep(1)
    except KeyboardInterrupt:
        transcriber.stop_recording()
        print("\nRecording stopped.")

if __name__ == "__main__":
    main() 