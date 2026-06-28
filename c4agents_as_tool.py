#---------------------------------------------------------------------#
# Descriptoin               : Following code is coded to meauser the distance between two cities. 
#                           : Here we practice  agent as tools
# Writtent by               : Muhammad Abdul Qayum
# Date                      : December 11, 2025  
#=====================================================================#
# Updation                  :
#=====================================================================#
#
#  0. Importing the necessary libraries
import os, requests, math
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
    model="gemini-2.5-flash-lite",
    openai_client=external_client
)
####################################################
# Create a tool


# Define the tool (function) your agent can call
@function_tool
def measure_distance(city1: str, city2: str) -> dict:
    """
    Tool to measure distance between two cities using OpenStreetMap + Haversine.
    Returns distance in kilometers.
    """
    def get_coordinates(city_name):
        url = "https://nominatim.openstreetmap.org/search"
        params = {"q": city_name, "format": "json", "limit": 1}
        response = requests.get(url, params=params, headers={"User-Agent": "distance-agent"})
        data = response.json()
        if not data:
            raise ValueError(f"No coordinates found for {city_name}")
        return float(data[0]["lat"]), float(data[0]["lon"])

    def haversine(lat1, lon1, lat2, lon2):
        R = 6371  # Earth radius in km
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

    lat1, lon1 = get_coordinates(city1)
    lat2, lon2 = get_coordinates(city2)
    distance_km = haversine(lat1, lon1, lat2, lon2)

    return {"city1": city1, "city2": city2, "distance_km": round(distance_km, 2)}


# @function_tool
# def get_coordinates(city_name):
#         """You are an AI agent that get latitude/longitude points of city. """
#         url = "https://nominatim.openstreetmap.org/search"
#         params = {"q": city_name, "format": "json", "limit": 1}
#         response = requests.get(url, params=params, headers={"User-Agent": "distance-agent"})
#         data = response.json()
#         if not data:
#             raise ValueError(f"No coordinates found for {city_name}")
#         return float(data[0]["lat"]), float(data[0]["lon"])


# # Create another worker agent
# location_agent = Agent(
#     name="Coordination finder",
#     instructions="You are an AI agent that get latitude/longitude points of city.",
#     model=llm_model,
#     tools=[get_coordinates]
#)
# Create another worker agent
distance_calculator_agent = Agent(
    name="DistanceCalculatorAgent",
    instructions="You are an AI agent that writes and runs Python code to calculate the distance in KM between two latitude/longitude points.",
    model=llm_model,
    tools=[measure_distance]
)

# Create the orchestrator agent
agent = Agent(
    name="Agent",
    instructions="You are an AI agent that calculates the distance between two locations. Use the Location Agent to get the latitude / longitude. Use the Distance Calculator agent to calculate the distance.",
    model=llm_model,
    tools=[
    # location_agent.as_tool(
    #     tool_name="LocationAgent",
    #     tool_description="Returns the latitude and longitude for a particular location"
    #     ),
    distance_calculator_agent.as_tool(
        tool_name="DistanceCalculatorAgent",
        tool_description="Calculates the distance between two latitude/longitude points"
        )]
)
result = Runner.run_sync(agent, "What's the straight-line distance between Karachi and Lahore?")
print(result.final_output)