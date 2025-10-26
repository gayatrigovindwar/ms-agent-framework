import asyncio
import os
from typing import Annotated
from dotenv import load_dotenv
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.google.google_ai import GoogleAIChatCompletion
from semantic_kernel.contents import ChatHistory
from semantic_kernel.functions import kernel_function
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior

load_dotenv()


class ResearchPlugin:
    """Plugin for research-related functions"""

    @kernel_function(
        name="search_information",
        description="Search for information on a given topic"
    )
    def search_information(
        self,
        topic: Annotated[str, "The topic to research"]
    ) -> Annotated[str, "Research findings"]:
        """Simulated research function"""
        research_data = {
            "python": "Python is a high-level, interpreted programming language known for its simplicity and readability. It was created by Guido van Rossum and first released in 1991.",
            "ai": "Artificial Intelligence (AI) refers to the simulation of human intelligence in machines. It includes machine learning, natural language processing, and computer vision.",
            "quantum computing": "Quantum computing uses quantum-mechanical phenomena like superposition and entanglement to perform operations on data. It has the potential to solve certain problems much faster than classical computers."
        }

        topic_lower = topic.lower()
        for key, value in research_data.items():
            if key in topic_lower:
                return value

        return f"Research on '{topic}': This is a complex topic that requires further investigation."

class DataAnalysisPlugin:
    """Plugin for data analysis functions"""

    @kernel_function(
        name="analyze_data",
        description="Analyze data and provide insights"
    )
    def analyze_data(
        self,
        data_description: Annotated[str, "Description of the data to analyze"]
    ) -> Annotated[str, "Analysis results"]:
        """Simulated data analysis function"""
        return f"Analysis of '{data_description}': The data shows a positive trend with a 15% increase over the previous period. Key insights include improved performance metrics and user engagement."


class WritingPlugin:
    """Plugin for content writing functions"""

    @kernel_function(
        name="create_content",
        description="Create written content based on topic and style"
    )
    def create_content(
        self,
        topic: Annotated[str, "The topic to write about"],
        style: Annotated[str, "Writing style (e.g., professional, casual, technical)"]
    ) -> Annotated[str, "Generated content"]:
        """Simulated content creation function"""
        return f"[{style.upper()} CONTENT]\n\n{topic}\n\nThis is a well-crafted piece of content tailored to the {style} style, covering the key aspects of {topic} with appropriate depth and clarity."


class Agent():
    def __init__(self, name: str, role: str, system_message: str):
        self.name = name
        self.role = role
        self.system_message = system_message
        self.kernel = Kernel()
        self.chat_history = ChatHistory()
        self.chat_history.add_system_message(system_message)

    async def initialize(self, gemini_api_key: str, service_id: str = "gemini-chat"):
        self.kernel.add_service(
            GoogleAIChatCompletion(
                service_id=service_id,
                gemini_model_id="gemini-2.5-flash",
                api_key=gemini_api_key
            )
        )
        self.service_id = service_id
        self.chat_service = self.kernel.get_service(service_id)

    def add_plugin(self, plugin, plugin_name:str):
        self.kernel.add_plugin(plugin, plugin_name)


    async def process(self, message: str) -> str:
        self.chat_history.add_user_message(message)

        execution_settings = self.kernel.get_prompt_execution_settings_from_service_id(self.service_id)
        execution_settings.function_choice_behavior = FunctionChoiceBehavior.Auto()

        response = await self.chat_service.get_chat_message_content(
            chat_history=self.chat_history,
            settings=execution_settings,
            kernel=self.kernel
        )

        self.chat_history.add_assistant_message(str(response))
        return str(response)
    

class Orchestrator():
    def __init__(self):
        self.agents = {}

    def add_agent(self, agent: Agent):
        self.agents[agent.name] = agent

    def get_agent(self, name: str) -> Agent:
        return self.agents.get(name)

    def list_agents(self):
        return [{"name": agent.name, "role": agent.role} for agent in self.agents.values()]


async def main():

    google_api_key = os.getenv("GOOGLE_API_KEY")

    orchestrator = Orchestrator()

    research_agent = Agent(
        name="ResearchAgent",
        role="Research Specialist",
        system_message="You are a research specialist. Your job is to gather and summarize information on various topics. Use the search_information tool when needed."
    )
    await research_agent.initialize(google_api_key)
    research_agent.add_plugin(ResearchPlugin(), "ResearchPlugin")
    orchestrator.add_agent(research_agent)

    analyst_agent = Agent(
        name="AnalystAgent",
        role="Data Analyst",
        system_message="You are a data analyst. Your job is to analyze data and provide insights. Use the analyze_data tool when needed."
    )
    await analyst_agent.initialize(google_api_key)
    analyst_agent.add_plugin(DataAnalysisPlugin(), "DataAnalysisPlugin")
    orchestrator.add_agent(analyst_agent)

    writer_agent = Agent(
        name="WriterAgent",
        role="Content Writer",
        system_message="You are a professional content writer. Your job is to create well-written content on various topics. Use the create_content tool when needed."
    )
    await writer_agent.initialize(google_api_key)
    writer_agent.add_plugin(WritingPlugin(), "WritingPlugin")
    orchestrator.add_agent(writer_agent)


    coordinator_agent = Agent(
        name="Coordinator",
        role="Task Coordinator",
        system_message=(
            "You are a coordinator that routes tasks to specialized agents. "
            "You have access to three agents:\n"
            "1. ResearchAgent - for research and information gathering\n"
            "2. AnalystAgent - for data analysis\n"
            "3. WriterAgent - for content creation\n\n"
            "When a user asks a question, determine which agent(s) should handle it and explain your routing decision."
        )
    )
    await coordinator_agent.initialize(google_api_key)
    orchestrator.add_agent(coordinator_agent)

    current_agent = coordinator_agent


    while True:
        user_input = input(f"You (talking to {current_agent.name}): ").strip()

        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        if user_input.startswith("switch:"):
            agent_name = user_input.split(":")[1].strip()
            agent = orchestrator.get_agent(agent_name)
            if agent:
                current_agent = agent
                print(f"✓ Switched to {agent_name}\n")
            else:
                print(f"✗ Agent '{agent_name}' not found\n")
            continue

        if not user_input:
            continue

        # Process message with current agent
        print(f"\n🤖 {current_agent.name}: ", end="", flush=True)
        response = await current_agent.process(user_input)
        print(f"{response}\n")

if __name__ == "__main__":
    asyncio.run(main())
