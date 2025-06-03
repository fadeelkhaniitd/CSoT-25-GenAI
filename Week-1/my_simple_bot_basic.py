import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def run_chat_agent():
    """
    Simulates a chatbot agent using a while loop and LLM API calls.
    """

    print("Welcome to the chatbot! Type 'exit' or 'quit' to end the conversation.")
    chat = gemini_client.chats.create(model="gemini-2.5-flash-preview-05-20")

    while True:
        user_input = input("You: ")
        # Check for exit conditions
        if user_input.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
        # Validate user input
        if not user_input.strip():
            print("Please enter something.")
            continue

        response = chat.send_message(user_input)
        print(f"Bot: {response.text}")


run_chat_agent()