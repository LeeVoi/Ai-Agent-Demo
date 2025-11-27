import os
import re
from dotenv import load_dotenv
from autogen import AssistantAgent, UserProxyAgent

# Load API key
load_dotenv()
API_KEY = os.getenv("MISTRAL_API_KEY")

# -----------------------------------------------------------
# LLM CONFIGURATION
# -----------------------------------------------------------
LLM_CONFIG = {
    "config_list": [
        {
            "model": "open-mistral-nemo",
            "api_key": API_KEY,
            "api_type": "mistral",
            "api_rate_limit": 0.25,
            "repeat_penalty": 1.1,
            "temperature": 0.0,
            "seed": 42,
            "stream": False,
            "native_tool_calls": False,
            "cache_seed": None,
        }
    ]
}

# -----------------------------------------------------------
# FAKE RESEARCH PAPER DATABASE
# -----------------------------------------------------------
PAPERS = [
    {"title": "Adaptive Meta-Learning Networks", "topic": "AI", "year": 2020, "citations": 340},
    {"title": "Efficient Reinforcement Agents with Sparse Rewards", "topic": "AI", "year": 2017, "citations": 780},
    {"title": "Probabilistic Graph Models for Real-Time Reasoning", "topic": "AI", "year": 2015, "citations": 530},

    {"title": "Quantum Annealing for Large-Scale Optimization", "topic": "Quantum Computing", "year": 2019, "citations": 215},
    {"title": "Cryogenic Qubit Stabilization Techniques", "topic": "Quantum Computing", "year": 2016, "citations": 150},
    {"title": "Distributed Quantum Circuits over Classical Networks", "topic": "Quantum Computing", "year": 2022, "citations": 30},

    {"title": "Deep Stereo Reconstruction in Dynamic Scenes", "topic": "Computer Vision", "year": 2020, "citations": 410},
    {"title": "Hierarchical Scene Understanding for Autonomous Vehicles", "topic": "Computer Vision", "year": 2017, "citations": 620},
    {"title": "Multi-Modal Object Detection with Thermal Fusion", "topic": "Computer Vision", "year": 2021, "citations": 150},

    {"title": "Federated Learning for Medical Diagnostics", "topic": "Machine Learning", "year": 2019, "citations": 280},
    {"title": "Uncertainty Calibration in Deep Neural Networks", "topic": "Machine Learning", "year": 2018, "citations": 670},
    {"title": "Low-Power Edge Models with On-Device Adaptation", "topic": "Machine Learning", "year": 2022, "citations": 55},

    {"title": "Blockchain-Based Secure Identity Systems", "topic": "Security", "year": 2018, "citations": 325},
    {"title": "Adaptive Intrusion Detection with Neural Signatures", "topic": "Security", "year": 2020, "citations": 190},
    {"title": "Post-Quantum Encryption Using Structured Lattices", "topic": "Security", "year": 2021, "citations": 140},
]

def search_papers(topic: str, comparator: str, year: int, citations: int):
    """Search the fake research paper database."""
    results = []
    for p in PAPERS:

        if topic.lower() not in p["topic"].lower():
            continue

        valid_year = (
            (comparator == "before" and p["year"] < year) or
            (comparator == "after" and p["year"] > year) or
            (comparator == "in" and p["year"] == year)
        )

        if not valid_year:
            continue

        if p["citations"] >= citations:
            results.append(p)

    return results


# -----------------------------------------------------------
# TOOL FUNCTION
# -----------------------------------------------------------
def paper_search_tool(topic: str, comparator: str, year: int, citations: int):
    """Wrapper for the research paper search function."""
    return search_papers(topic, comparator, year, citations)


# -----------------------------------------------------------
# ASSISTANT AGENT SETUP
# -----------------------------------------------------------
assistant = AssistantAgent(
    "assistant",
    llm_config=LLM_CONFIG,
    system_message=(
        "You MUST respond ONLY by calling the function 'paper_search_tool'. "
        "Never explain. Never write natural language sentences. "
        "Always output exactly this format:\n\n"
        "paper_search_tool(topic='AI', comparator='before', year=2020, citations=100)\n\n"
        "Replace values based on the user's request."
    ),
)

assistant.register_function({
    "paper_search_tool": paper_search_tool
})


# -----------------------------------------------------------
# TOOL CALL INTERPRETER
# -----------------------------------------------------------
def try_execute_tool(model_output: str):
    """Detect and execute paper_search_tool(...) from model output."""

    match = re.search(r"paper_search_tool\((.*?)\)", model_output)
    if not match:
        return None

    arg_string = match.group(1)
    args = dict(re.findall(r"(\w+)\s*=\s*'?(.*?)'?(?=,|\)|$)", arg_string))

    topic = args.get("topic", "").strip("'\"")
    comparator = args.get("comparator", "in").strip("'\"")

    # Safe integer conversions
    try:
        year = int(args.get("year", 0))
    except ValueError:
        year = 0

    try:
        citations = int(args.get("citations", 0))
    except ValueError:
        citations = 0

    return paper_search_tool(topic, comparator, year, citations)


# -----------------------------------------------------------
# INTERACTIVE LOOP
# -----------------------------------------------------------
def main():
    print("Research Paper Agent (type 'exit' to quit)\n")

    last_assistant_call = None
    last_tool_result = None

    while True:
        user_query = input("You: ")

        if user_query.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        # ---------------------------------------------------------
        # EVALUATION REQUEST
        # ---------------------------------------------------------
        if user_query.lower().startswith("evaluate"):
            if last_assistant_call is None:
                print("\nNo previous task to evaluate.\n")
                continue

            # Build evaluation context
            eval_prompt = (
                "Evaluate how well you performed on the previous task.\n\n"
                f"Assistant Tool Call: {last_assistant_call}\n"
                f"Tool Result: {last_tool_result}\n\n"
                "Discuss correctness, tool usage, and give a score from 1 to 10."
            )

            response = assistant.generate_reply(
                messages=[{"role": "user", "content": eval_prompt}]
            )

            print("\nEvaluation:\n", response.get("content", ""))
            print()
            continue
        # ---------------------------------------------------------

        # Normal query to the assistant
        response = assistant.generate_reply(
            messages=[{"role": "user", "content": user_query}]
        )

        assistant_text = response.get("content", "")
        print("\nAssistant:", assistant_text)

        # Run tool
        tool_result = try_execute_tool(assistant_text)

        if tool_result is not None:
            if len(tool_result) > 0:
                print("Tool executed:", tool_result)
            else:
                print("Tool executed: No papers matched your query.")

        # Save for evaluation
        last_assistant_call = assistant_text
        last_tool_result = tool_result

        print()



if __name__ == "__main__":
    main()
