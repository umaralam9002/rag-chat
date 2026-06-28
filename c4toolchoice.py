# #  0. Importing the necessary libraries
# import os
# from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI, set_tracing_disabled, function_tool, StopAtTools
# from dotenv import load_dotenv, find_dotenv
# # 0.1. Loading the environment variables
# load_dotenv(find_dotenv())
# set_tracing_disabled(disabled=True)
# # 1. Which LLM Provider to use? -> Google Chat Completions API Service
# external_client: AsyncOpenAI = AsyncOpenAI(
#     api_key=os.getenv("GEMINI_API_KEY"),
#     base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
# )

# # 2. Which LLM Model to use?
# llm_model: OpenAIChatCompletionsModel = OpenAIChatCompletionsModel(
#     model="gemini-2.5-flash-lite",
#     openai_client=external_client
# )
# ####################################################
# # Create a tool
# @function_tool
# def create_invoice(orderID: int) -> str:
#  return f"Invoice for Order {orderID}: $123.45 (Generated on 2025-07-05)"

# # 3. Creating the Agent
# agent = Agent(
#     name="Invoice Generator agent",
#     instructions="Generate and return the invoice when requested",
#     model=llm_model,
#     tools=[create_invoice], tool_use_behavior=StopAtTools(stop_at_tool_names=["create_invoice"]),
# )


# result = Runner.run_sync(agent,
#     input="Please create an invoice for order ID 200",
#     #stop=StopAtTools(names=["create_invoice"])  # stop once tool is reached
# )

# print("AGENT RESPONSE: " , result.final_output)


import os
from agents import (
    Agent,
    Runner,
    OpenAIChatCompletionsModel,
    AsyncOpenAI,
    set_tracing_disabled,
    function_tool,
    StopAtTools,
)
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv())
set_tracing_disabled(disabled=True)

# LLM Client
external_client = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Model
llm_model = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash-lite",
    openai_client=external_client
)

# Tool
@function_tool
def create_invoice(orderID: int) -> str:
    return f"Invoice for Order {orderID}: $123.45 (Generated on 2025-07-05)"

# Agent
agent = Agent(
    name="Invoice Generator Agent",
    instructions="Generate and return the invoice when requested.",
    model=llm_model,
    tools=[create_invoice],
    tool_use_behavior=StopAtTools(
        stop_at_tool_names=["create_invoice"]
    ),
)

print("Hello! What can I help you with?")
print("Type 'exit' to quit.\n")

# Chat Loop
while True:
    user_input = input("You: ")

    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    result = Runner.run_sync(
        agent,
        input=user_input
    )

    print("Agent:", result.final_output)