import os
from dotenv import load_dotenv, find_dotenv

from agents import (
    Agent,
    Runner,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    function_tool,
    set_tracing_disabled,
)

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv(find_dotenv())
set_tracing_disabled(disabled=True)

# -----------------------------
# Gemini OpenAI-Compatible Client
# -----------------------------
client = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# -----------------------------
# Model
# -----------------------------
model = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash-lite",
    openai_client=client,
)

# -----------------------------
# Mortgage Calculator Tool
# -----------------------------
@function_tool
def calculate_mortgage(
    principal_amount: float,
    annual_rate: float,
    years: int,
) -> str:
    """
    Calculate the monthly mortgage payment.

    Args:
        principal_amount: Total loan amount.
        annual_rate: Annual interest rate in percent.
        years: Loan duration in years.

    Returns:
        Monthly mortgage payment.
    """

    monthly_rate = (annual_rate / 100) / 12
    total_months = years * 12

    payment = (
        principal_amount
        * monthly_rate
        / (1 - (1 + monthly_rate) ** (-total_months))
    )

    return f"Monthly mortgage payment: ${payment:,.2f}"

# -----------------------------
# Agent
# -----------------------------
mortgage_agent = Agent(
    name="Mortgage Advisor",
    instructions="""
You are a mortgage assistant.

Whenever the user asks about a mortgage payment,
always use the calculate_mortgage tool.
After receiving the tool result, explain it briefly.
""",
    model=model,
    tools=[calculate_mortgage],
)

# -----------------------------
# Run Agent
# -----------------------------
result = Runner.run_sync(
    mortgage_agent,
    input="What is my monthly payment if I borrow $800,000 at 6% interest for 30 years?"
)

print(result.final_output)