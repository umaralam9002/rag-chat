#  0. Importing the necessary libraries
import os
from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI, set_tracing_disabled, function_tool
from dotenv import load_dotenv, find_dotenv
# 0.1. Loading the environment variables
load_dotenv(find_dotenv())
set_tracing_disabled(disabled=True)
# 1. Which LLM Provider to use? -> Google Chat Completions API Service
external_client: AsyncOpenAI = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# 2. Which LLM Model to use?
llm_model: OpenAIChatCompletionsModel = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=external_client
)
# Create a tool
@function_tool
def get_order_status(orderID: int) -> str:
    """
    Returns the order status given an order ID
    Args:
    orderID (int) - Order ID of the customer's order
    Returns:
    string - Status message of the customer's order
    """
    if orderID in (100, 101):
        return "Delivered"
    elif orderID in (200, 201):
        return "Delayed"
    elif orderID in (300, 301):
        return "Cancelled"

# 3. Creating the Agent
agent = Agent(name="Customer service agent",
    instructions="You are an AI Agent that helps respond to customer queries for a local paper company", 
    tools=[get_order_status], model=llm_model)

# 4. Running the Agent
result = Runner.run_sync(agent,
 input="What's the status of my order? My Order ID is 100"
 ) 

print("AGENT RESPONSE: " , result.final_output)