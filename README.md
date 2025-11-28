# Research Paper AI Agent (Autogen + Mistral)

This project implements an AI agent for **Compulsory Assignment #2: AI Agents Using Autogen**.

The goal of the assignment is to create an agent that can answer queries such as:

**“Find a research paper on [topic] published [before/after/in] [year] with at least [citations] citations.”**

The agent must solve the task using a **tool**, not normal LLM text responses.

---

## How It Works

### 1. Autogen Assistant Agent  
The agent is configured with a strict system message that forces it to respond **only** by calling a function named `paper_search_tool(...)`.

### 2. Tool Function  
`paper_search_tool` searches a small **fake research paper dataset** included in the project.  
It allows the AI Agent to filter papers based on:
- topic  
- comparator (before / after / in)  
- year  
- minimum citations  

### 3. Tool Call Execution  
The model outputs tool calls as text.  
A small parser detects these calls, extracts the arguments, and runs the Python function.

### 4. Interactive CLI  
Running the program gives a simple terminal interface:

