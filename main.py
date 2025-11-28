import os
import re
from dotenv import load_dotenv
from autogen import AssistantAgent, UserProxyAgent

# Load environment variables (used for loading the Mistral API key)
load_dotenv()
API_KEY = os.getenv("MISTRAL_API_KEY")

# -----------------------------------------------------------
# LLM CONFIGURATION
# -----------------------------------------------------------
# This configuration comes directly from the assignment requirements.
# It defines which cloud LLM to use, how to authenticate, and how
# Autogen should handle request rate limits and tool call behavior.
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
# Instead of using a live API, we implement a mock dataset.
# The agent will query this "database" using its tool.
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
    """
    Search the fake paper database using simple filtering logic.

    - Topic must match (case-insensitive substring)
    - Year comparison uses "before", "after", or "in"
    - Citations must meet the minimum threshold
    """
    results = []
    for p in PAPERS:

        # Topic filtering
        if topic.lower() not in p["topic"].lower():
            continue

        # Compare the publication year using the comparator
        valid_year = (
            (comparator == "before" and p["year"] < year) or
            (comparator == "after" and p["year"] > year) or
            (comparator == "in" and p["year"] == year)
        )
        if not valid_year:
            continue

        # Citation count filtering
        if p["citations"] >= citations:
            results.append(p)

    return results


# -----------------------------------------------------------
# TOOL FUNCTION
# -----------------------------------------------------------
# This is the tool the model will call.
# Autogen enforces that the Assistant must call this function.
def paper_search_tool(topic: str, comparator: str, year: int, citations: int):
    """Wrapper for the research paper search function."""
    return search_papers(topic, comparator, year, citations)


# -----------------------------------------------------------
# ASSISTANT AGENT SETUP
# -----------------------------------------------------------
# The system message forces the LLM to respond ONLY with the tool call.
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

# Registering the tool
assistant.register_function({
    "paper_search_tool": paper_search_tool
})


# -----------------------------------------------------------
# TOOL CALL INTERPRETER
# -----------------------------------------------------------
# Autogen (in this fork) outputs tool calls as plaintext.
# This function extracts the tool arguments and executes it.
def try_execute_tool(model_output: str):
    """Detect and execute paper_search_tool(...) from model output."""

    # Match the function call text using regex
    match = re.search(r"paper_search_tool\((.*?)\)", model_output)
    if not match:
        return None

    arg_string = match.group(1)

    # Extract key=value pairs from the argument list
    args = dict(re.findall(r"(\w+)\s*=\s*'?(.*?)'?(?=,|\)|$)", arg_string))

    # Extract string fields
    topic = args.get("topic", "").strip("'\"")
    comparator = args.get("comparator", "in").strip("'\"")

    # Convert numeric fields safely
    try:
        year = int(args.get("year", 0))
    except ValueError:
        year = 0

    try:
        citations = int(args.get("citations", 0))
    except ValueError:
        citations = 0

    # Call the tool
    return paper_search_tool(topic, comparator, year, citations)


# -----------------------------------------------------------
# INTERACTIVE LOOP
# -----------------------------------------------------------
# Provides a command-line interface to ask questions,
# trigger evaluation, and exit the program.
def main():
    print("Research Paper Agent (type 'exit' to quit)\n")

    last_assistant_call = None
    last_tool_result = None

    while True:
        # Get user input
        user_query = input("You: ")
        last_user_query = user_query

        # Exit the program
        if user_query.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        # ---------------------------------------------------------
        # EVALUATION REQUEST
        # ---------------------------------------------------------
        # If the user types "evaluate", the LLM acts as a judge
        if user_query.lower().startswith("evaluate"):
            if last_assistant_call is None:
                print("\nNo previous task to evaluate.\n")
                continue

            # A structured evaluation prompt using best practices
            eval_prompt = f"""
            You are evaluating whether the AGENT'S TOOL RESULT correctly satisfied the USER'S QUERY.

            To improve your reasoning, you must follow this evaluation structure:
            - First, think step-by-step inside the 'Evaluation:' field.
            - Then, provide a single integer rating from 1 to 4 inside 'Rating:'.

            ---------------------------------------
            USER QUERY:
            {last_user_query}

            TOOL CALL:
            {last_assistant_call}

            TOOL RESULT:
            {last_tool_result}
            ---------------------------------------

            ### IMPORTANT RULES (FOLLOW STRICTLY):
            1. "after YEAR"  means publication_year > YEAR
            2. "before YEAR" means publication_year < YEAR
            3. "in YEAR"     means publication_year == YEAR
            4. Citation count must be AT LEAST the requested number.
            5. Topic must appear inside the paper's topic (case-insensitive).
            6. If multiple papers are returned, all must satisfy the constraints.
            7. Do NOT invent logical mistakes.
            8. Do NOT penalize results for having EXTRA citations or being AFTER the year unless that violates the comparator.

            ### RATING SCALE (1–4):
            1 = Completely incorrect: violates major constraints, irrelevant or wrong.
            2 = Partially incorrect: some constraints satisfied, but important ones violated.
            3 = Mostly correct: satisfies most constraints, minor issues only.
            4 = Excellent: perfectly satisfies ALL constraints.

            ### OUTPUT FORMAT (YOU MUST FOLLOW EXACTLY):
            Feedback:::
            Evaluation: <your reasoning here>
            Rating: <integer 1–4>

            If your rating is correct, you will receive 100 H100 GPUs to start your AI company.

            Now, begin your evaluation.
            Feedback:::
            Evaluation:
            """

            # Ask the model for the evaluation
            response = assistant.generate_reply(
                messages=[{"role": "user", "content": eval_prompt}]
            )

            print("\nEvaluation:\n", response.get("content", ""))
            print()
            continue

        # ---------------------------------------------------------
        # NORMAL USER QUERY (TOOL CALL GENERATION)
        # ---------------------------------------------------------
        # The assistant converts the query → tool call
        response = assistant.generate_reply(
            messages=[{"role": "user", "content": user_query}]
        )

        assistant_text = response.get("content", "")
        print("\nAssistant:", assistant_text)

        # Execute the tool call if one was produced
        tool_result = try_execute_tool(assistant_text)

        if tool_result is not None:
            if len(tool_result) > 0:
                print("Tool executed:", tool_result)
            else:
                print("Tool executed: No papers matched your query.")

        # Save values for possible evaluation
        last_assistant_call = assistant_text
        last_tool_result = tool_result

        print()  # Formatting spacer


if __name__ == "__main__":
    main()
