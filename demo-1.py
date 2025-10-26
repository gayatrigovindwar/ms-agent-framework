import asyncio
import os

from dotenv import load_dotenv
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.google.google_ai import GoogleAIChatCompletion
from semantic_kernel.contents import ChatHistory

load_dotenv()

async def main():

    kernel = Kernel()

    gemini_api_key = os.getenv("GOOGLE_API_KEY")

    kernel.add_service(
        GoogleAIChatCompletion(
            service_id="gemini",
            gemini_model_id="gemini-2.5-flash",
            api_key=gemini_api_key
        )
    )

    chat_history = ChatHistory()
    chat_history.add_system_message(
        "You are Python developer and your name Alex, you will only answers queries python."
    )

    chat_service = kernel.get_service("gemini")


    while True:

        user_input = input("User: ").strip()

        if user_input.lower() == "exit":
            print("bye")
            break

        if not user_input:
            continue

        chat_history.add_user_message(user_input)

        response =  await chat_service.get_chat_message_content(
            chat_history = chat_history,
            settings = kernel.get_prompt_execution_settings_from_service_id("gemini")
        )

        chat_history.add_assistant_message(str(response))

        print(f"Agent: {response}")


if __name__ == "__main__":
    asyncio.run(main())