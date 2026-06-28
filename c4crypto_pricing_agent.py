#  0. Importing the necessary libraries
import os, requests
from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI, set_tracing_disabled, function_tool, StopAtTools
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
    model="gemini-2.5-flash-lite",
    openai_client=external_client
)
####################################################
# Create a tool
@function_tool
def get_price_of_bitcoin() -> str:
    """Get the price of Bitcoin."""
    url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"
    response = requests.get(url)
    price = response.json()["bitcoin"]["usd"]
    return f"${price:,.2f} USD."

# 3.# Define an agent that u
crypto_agent = Agent(
    name="CryptoTracker",
    instructions="You are a crypto assistant. Use tools to get real-time data.",
    model=llm_model,
    tools=[get_price_of_bitcoin]
    )
# Run the agent with an example question
result = Runner.run_sync(crypto_agent, "What's the price of Bitcoin?")
print(result.final_output)