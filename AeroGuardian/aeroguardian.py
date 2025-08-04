# AeroGuardian: Agentic AI System Using LangGraph
# Modern aircraft rely on large volumes of sensor data and pilot feedback to ensure safe operation.
# However, in real-world scenarios, unexpected anomalies like engine overheating, fuel imbalance, or severe turbulence can arise mid-flight.
# AeroGuardian aims to simulate an intelligent multi-agent co-pilot that:

# -> Analyzes incoming (simulated) flight data logs ---> [perception_node],

# -> Compares it against previous flight incidents ---> [RAG - future scope]

# -> Decides whether the current situation is safe or risky ---> [decision_node],

# -> Suggests corrective actions  --->  [suggestion_agent - LLM call]

# -> Updates the current state according to the actions suggested by AI Agent (LLM) ---> [apply_corrective_actions function]

# -> Sends back this correctly updated state to "perception_node" -> "decision_node" -> if safe? "log_node" -> END

# -> Exits the loop safely when conditions stabilize.

# Implemented using LangGraph to orchestrate intelligent decision-making across agents with loops and conditional transitions.


# ---------------------------- ALL IMPORTS --------------------------------------------------------------------------------

from langgraph.graph import StateGraph, END
from langchain.schema import SystemMessage
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableLambda
import random

import os
from dotenv import load_dotenv

'''
For Ollama models : 
uv pip install -U langchain-ollama   (For below 2 lines of import)
'''
from langchain_ollama import OllamaLLM
from langchain_openai import ChatOpenAI


from typing import Optional, Dict, Any, TypedDict
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode

# ----------------------------------------------------------------------------------------------

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# --------------------- DECIDING MODEL ------------------------------------------------------------

# MODEL = 'gpt-3.5-turbo'
MODEL = 'llama3.1'


# @tool
# def tool1():
#     """Don't forget to add a docstring. It's must to mention"""
#     # pass

# @tool
# def tool2():
#     """Don't forget to add a docstring. It's must to mention"""
#     # pass

# tools = [tool1, tool2]


TEMPERATURE = 0.7
if MODEL.startswith('gpt'):
    model = ChatOpenAI(api_key=OPENAI_API_KEY, model=MODEL)
else:
    model = OllamaLLM(model=MODEL, temperature=TEMPERATURE)
    # model = OllamaLLM(model=MODEL, temperature=TEMPERATURE).bind_tools(tools) # I don't have any need to bind any tools here, as I don't have any at the moment.


# response = model.invoke("Tell me a joke")
# print(response)


# -----------------------------------------------------------------------------------------------



class AgentState(TypedDict):
    current_data: Optional[Dict[str, Any]]
    features: Optional[Dict[str, str]]
    risk_score: Optional[int]
    status: Optional[str]
    action: Optional[str]




# Step 2: Simulated Input Generator (Aircraft Data)
def simulate_flight_data():
    return {
        "altitude": random.choice([30000, 32000, 34000, 36000]),
        "engine_temp": random.randint(850, 950),
        "fuel_level": round(random.uniform(30.0, 50.0), 1),
        "weather": random.choice(["clear", "moderate turbulence", "storm"]),
        "pilot_report": random.choice([
            "normal",
            "slight vibration in engine",
            "noticeable rumble",
            "vibration increasing",
        ])
    }


# Step 2B: Apply Corrective Actions to Flight Data - Pure Function
def apply_corrective_actions(current_data, action_text):
    """Apply LLM suggested corrective actions to modify flight data"""
    modified_data = current_data.copy()
    action_lower = action_text.lower()
    
    # Apply engine temperature corrections
    if "reduce engine power" in action_lower or "cool engine" in action_lower or "lower engine" in action_lower:
        modified_data["engine_temp"] = max(850, modified_data["engine_temp"] - random.randint(30, 60))
    
    # Apply fuel corrections
    if "refuel" in action_lower or "increase fuel" in action_lower or "fuel management" in action_lower:
        modified_data["fuel_level"] = min(50.0, modified_data["fuel_level"] + random.uniform(5.0, 15.0))
    
    # Apply altitude/weather corrections
    if "change altitude" in action_lower or "descend" in action_lower or "lower altitude" in action_lower:
        modified_data["altitude"] = random.choice([28000, 30000, 32000])
        # Changing altitude often helps with weather
        if modified_data["weather"] in ["moderate turbulence", "storm"]:
            modified_data["weather"] = random.choice(["clear", "moderate turbulence"])
    
    # Apply pilot report improvements
    if "monitor" in action_lower or "check" in action_lower or "reduce vibration" in action_lower:
        if "vibration" in modified_data["pilot_report"]:
            modified_data["pilot_report"] = random.choice(["normal", "slight vibration in engine"])
    
    print(f"🔧 Applied corrective actions to data:")
    print(f"   Original: {current_data}")
    print(f"   Modified: {modified_data}")
    
    return modified_data




# Step 3: Perception Node - Pure Function (Feature Extraction)
def perception_node(state: AgentState) -> AgentState:

    if state.get("current_data") and state.get("action") and state.get("status") == "risky":
        # Apply corrective actions to existing data
        input_data = apply_corrective_actions(state["current_data"], state["action"])
    else:
        input_data = simulate_flight_data()

    # Example extracted features:
    features = {
        "temp_status": "high" if input_data["engine_temp"] > 900 else "normal",
        "fuel_status": "low" if input_data["fuel_level"] < 35 else "normal",
        "weather_severity": "bad" if input_data["weather"] != "clear" else "good",
        "pilot_alert": "yes" if "vibration" in input_data["pilot_report"] else "no"
    }

    # Update state dictionary
    state["current_data"] = input_data  
    state["features"] = features

    return state



# Step 4: Decision Node - Pure Function (Risk Evaluation)
def decision_node(state: AgentState) -> AgentState:
    features = state.get("features", {})
    risk_score = 0
    
    if features.get("temp_status") == "high": risk_score += 2
    if features.get("fuel_status") == "low": risk_score += 3
    if features.get("weather_severity") == "bad": risk_score += 1
    if features.get("pilot_alert") == "yes": risk_score += 2

    status = "risky" if risk_score >= 4 else "safe"

    state["risk_score"] = risk_score
    state["status"] = status
    
    return state




# Step 5A: Suggestion Agent (if RISKY)
def suggestion_agent(state: AgentState) -> AgentState:
    # Generate human-like recommendation using LLM

    print("\n🚨 RISK DETECTED - Generating suggestions...")
    print("CURRENT DATA -> ", state["current_data"])

    prompt = PromptTemplate.from_template(
        """You have to analyse the type of risk, based on the flight conditions below. 
        If the status is risky, you must suggest corrective actions
        to get back the stability of our flight.
        The corrective actions must be aligned with the Data and Features provided.
        
        Data: {data}
        Features: {features}
        Risk Score: {risk}
        Status: {overall_status}

        You have to try reducing the risk score, in order to get back the status to safe.
        Risk score is calculated based on the feature values:
        if "temp_status" in features is "high", then risk_score is increased by 2.
        if "fuel_status" in features is "low", then risk_score is increased by 3.
        if "weather_severity" in features is "bad", then risk_score is increased by 1.
        if "pilot_alert" in features is "yes", then risk_score is increased by 2.

        Risk score greater than 4 is considered as high risk.

        VERY IMPORTANT: Your response must include specific action keywords for the system to execute:
        - For high engine temperature: include "reduce engine power" or "cool engine"
        - For low fuel: include "refuel" or "increase fuel"
        - For bad weather/turbulence: include "change altitude" or "descend"
        - For vibration issues: include "monitor" or "reduce vibration"

        The final answer should be concise, STRICTLY NOT more than 150 words and include the exact action keywords above.
        """
    )
    chain = prompt | model

    response = chain.invoke({
        "data": state["current_data"],
        "features": state["features"],
        "risk": state["risk_score"],
        "overall_status": state["status"]
    })

    print("response ========> ", response)

    # Handle different response types (string for Ollama, object with content for OpenAI)
    if hasattr(response, 'content'):
        action_text = response.content
    else:
        action_text = str(response)
    
    state["action"] = action_text
    return state


# Step 5B: Log Node (if SAFE)
def log_node(state: AgentState) -> AgentState:
    print("\n✅ SAFE: Logging flight data...")
    print("Data:", state["current_data"])
    return state




# Step 6: Define Graph with Loop + Conditional Edge
graph = StateGraph(AgentState)
graph.add_node("perceive", perception_node)
graph.add_node("decide", decision_node)
graph.add_node("suggest", suggestion_agent)
graph.add_node("log", log_node)

# Edges
graph.set_entry_point("perceive")
graph.add_edge("perceive", "decide")
graph.add_conditional_edges(
    "decide",
    lambda state: state["status"],
    {
        "risky": "suggest",
        "safe": "log"
    }
)
graph.add_edge("suggest", "perceive")  # Loop back
graph.add_edge("log", END)       # NO Loop back

# graph.set_finish_point(END)
# graph.set_finish_point("log")
# graph.set_finish_point("suggest")



# Step 7: Compile and Run the Graph
app = graph.compile()

# from IPython.display import Image, display
# display(Image(graph.get_graph().draw_mermaid_png()))

# Get the PNG bytes
png_bytes = app.get_graph().draw_mermaid_png()

# Save to file
with open("graph.png", "wb") as f:
    f.write(png_bytes)

print("✅ Graph saved as graph.png")




# Run the system for 3 iterations (simulated loop)
state = {}
for i in range(3):

    print(f"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ITERATION {i+1} ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    result = app.invoke(state)

    status = result["status"]
    risk_score = result["risk_score"]
    current_data = result["current_data"]
    
    print("-"*25)
    print(f"Status: {status}")
    print(f"Risk Score: {risk_score}")
    print(f"Current Data: {current_data}")
    print("\n✅ Flight conditions stable - monitoring complete")
    print("-"*25)
    print()

    # Reset state for next iteration (simulate new data)
    initial_state = {
        "current_data": None,
        "features": None,
        "risk_score": None,
        "status": None,
        "action": None
    }

print("\n✅ AeroGuardian loop complete.")
