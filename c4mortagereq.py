#  0. Importing the necessary libraries
from typing import Any


import os
from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI, set_tracing_disabled, function_tool, ModelSettings, StopAtTools
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
def calculate_mortgage(
    principal_amount: float, annualized_rate: float, number_of_years: int
    ) -> str:
    """
    This function calculates the mortgage payment.
    Args:
    principal_amount: The mortgage amount.
    annual_rate: The annualized interest rate in percent form.
    years: The loan term in years.
    Returns:
    A message stating the monthly payment amount.
    """
    monthly_rate = (annualized_rate / 100) / 12
    months = number_of_years * 12
    payment = principal_amount * (monthly_rate) / (1 - (1 + monthly_rate) ** -months)
    print(payment)
    return f"${payment:,.2f}."

# 3.# Define an agent that uses the mortgage calculator tool
mortgage_agent = Agent[Any] (
    name="MortgageAdvisor",
    instructions=("You are a mortgage assistant"),
     model=llm_model,
    tools=[calculate_mortgage],
    tool_use_behavior="stop_on_first_tool",
    model_settings=ModelSettings(tool_choice="required" )
    )
# Run the agent with an example question
result = Runner.run_sync(mortgage_agent, "What is my monthly payments if I borrow $800,000 at 6% interest for 30 years?")

print(result.final_output)