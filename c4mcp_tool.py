#---------------------------------------------------------------------#
# Descriptoin               : Following progranm connect to mcp server. 
#                           : Here we practice  mcp
# Writtent by               : Muhammad Abdul Qayum
# Date                      : December 11, 2025  
#=====================================================================#
# Updation                  :
#=====================================================================#
#
#  0. Importing the necessary libraries
import os, asyncio
from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI, set_tracing_disabled
from dotenv import load_dotenv, find_dotenv
#from agents.tool import MCPServerSse, 
from agents.mcp import MCPServerSse

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
#---------------------------------------------------------------------------------------#
# Create the tool


async def main():
    # Create the MCP server (SSE transport). Note: pass URL inside params.
    async with MCPServerSse(params={"url": "https://mcp.api.coingecko.com/sse"}, name="coingecko") as mcp_server:
        # At this point, .connect() has already been called by the context manager.

        agent = Agent(
            name="Crypto Assistant",
            instructions="Use the CoinGecko MCP tools to answer questions.",
            model = llm_model,
            mcp_servers=[mcp_server],  # attach the already-connected server
        )

        result = await Runner.run(agent, "Get BTC price in USD and summarize 24h change.")
        print(result.final_output)

asyncio.run(main())




#agent = Agent(name="Git Assistant", mcp_servers=[crypt_mcp ])
#mcp_result = await Runner.run(agent, "Summarize failing checks on open PRs.")
#summary = mcp_result.final_output  # <-- tool outputs aggregated here


#-------------------------
# Create the agent
# agent = Agent(
#     name="Crypto Agent",
#     instructions="You are an AI agent that returns crypto prices.",
#     model = llm_model,
#     mcp_servers=[mcp_server]
# )
# result = Runner.run_sync(agent, "What's the price of bitcoin?")
# print(result.final_output)
