import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from groq import Groq
load_dotenv()
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def run_chat_agent():
    """
    Simulates a chatbot agent using a while loop and LLM API calls.
    """

    def groq_generate(prompt, system_prompt=None):
        if system_prompt:
            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "system", "content": system_prompt},
                          {"role": "user", "content": prompt}],
            )
        else:
            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
            )
        return response.choices[0].message.content


    def gemini_generate(prompt, system_prompt=None):
        if system_prompt:
            response = gemini_client.models.generate_content(
                model="gemini-2.5-flash-preview-05-20",
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt),
                contents=prompt
            )
        else:
            response = gemini_client.models.generate_content(
                model="gemini-2.5-flash-preview-05-20",
                contents=prompt
            )
        return response.text


    # Different system prompts used
    assistant_system_prompt = "You are a helpful assistant. You are provided the chat history, and you have to reply to the user's query."
    summarizer_system_prompt = "You are a chat history summarizer. You are provided the chat history in the format of 'User:[query]\\nResponder:[response]\\n...', and your job is to summarize it to preserve meaning while reducing length. IMPORTANT: KEEP the chat history format of User: and Responder:, and DO NOT include any other explanatory text."
    filterer_system_prompt = "You are a chat history filterer. You are provided 3 things, a query, an initial response to that query, and the chat history in the format of 'User:[query]\\nResponder:[response]\\n...'. Note that the chat history may be empty, or may not contain anything relevant to the query and response. In that case, do not generate anything on your own. Your job is to filter out chats from the CHAT HISTORY ONLY, which are relevant to the query and the initial response. IMPORTANT: KEEP the chat history format of User: and Responder:, DO NOT include any other explanatory text, and DO NOT INCLUDE the current query and response in the filtered chat history."
    reviewer_system_prompt = "You are a critical reviewer. You are provided 3 things, the relevant chat history in the format of 'User:[query]\\nResponder:[response]\\n...', the current user query, and the response provided by the responder. Your job is to review the response provided to the current query, and provide feedback, keeping the relevant chat history in mind."
    finalizer_system_prompt = "Your job is to provide the final response to a user query. You are given 4 things, the user's query, the initial response provided to the query, the feedback for the initial response, and the relevant chat history between the user and the responder in the format of 'User:[query]\\nResponder:[response]\\n...'"

    print("Welcome to the chatbot! Type 'exit' or 'quit' to end the conversation.")
    chat_history = ""

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

        # Summarize the chat history to reduce length
        chat_history = gemini_generate(chat_history, system_prompt=summarizer_system_prompt)
        if not chat_history: chat_history = "(empty)"

        # Generate initial response
        prompt = "Chat History:\n" + chat_history + "\n\nUser query: " + user_input
        initial_response = gemini_generate(prompt, system_prompt=assistant_system_prompt)

        # Filter out relevant chat history
        filterer_prompt = "Query: " + user_input + "\nInitial Response: " + initial_response + "\n\nChat History:\n" + chat_history
        filtered_chat_history = groq_generate(filterer_prompt, system_prompt=filterer_system_prompt)
        if not filtered_chat_history: filtered_chat_history = "(empty)"

        # Provide feedback on initial response
        reviewer_prompt = "Relevant Chat History:\n\n" + filtered_chat_history + "\n\nUser query: " + user_input + "\n\nResponder's response: " + initial_response
        feedback = groq_generate(reviewer_prompt, system_prompt=reviewer_system_prompt)

        # Generate final response based on feedback and relevant chat history
        final_prompt = "User query: " + user_input + "\n\nInitial response: " + initial_response + "\n\nFeedback: " + feedback + "\n\nRelevant Chat History:\n\n" + filtered_chat_history
        final_response = gemini_generate(final_prompt, system_prompt=finalizer_system_prompt)
        chat_history += f"\nUser: {user_input}\nResponder: {final_response}\n"

        print("Bot: ", final_response)


run_chat_agent()
