"""
ReAct Agent — Reasoning + Acting
==================================
A general-purpose agent architecture that interleaves chain-of-thought reasoning
with executable tool actions, enabling language models to solve multi-step tasks
by dynamically deciding what to do next.

Architecture:
    At each step the agent produces a Thought → Action → Observation triplet:
        Thought:     free-form reasoning about the current state and what to do next
        Action:      a structured tool call (name + arguments)
        Observation: the result returned by the tool
    The loop repeats until the agent emits a "Finish" action with the final answer.

Formal trace:
    s_0 = task
    For t = 1, 2, ..., T:
        Thought_t  = LM(prompt + history_{t-1})
        Action_t   = parse(Thought_t)
        Observation_t = tool(Action_t)
        history_t  = history_{t-1} + [Thought_t, Action_t, Observation_t]
    Return: Finish answer

Key insight (Yao et al., 2022):
    Keeping thoughts in the context window allows the model to self-correct:
    if an observation contradicts a prior assumption, the next Thought can revise
    the plan. Pure chain-of-thought has no feedback; pure action has no reasoning.

This implementation provides:
    - A rule-based "mock LLM" for demonstrating the loop without an API call
    - Tool registry with arithmetic, search, and state tools
    - Full ReAct loop with scratchpad history
    - Drop-in slot to swap in a real LLM backend (any hosted or local model API)

Use Cases:
    - Question-answering with tool use (retrieval, calculator, code execution)
    - Autonomous web research agents
    - Code generation with test-and-fix loops
    - Multi-step planning in simulated environments (ALFWorld, WebShop)

Advantages:
    - Simple: just prompt engineering + tool routing, no RL required
    - Interpretable: the full reasoning trace is human-readable
    - Flexible: add new tools without retraining the model
    - Self-correcting: observations feed back into reasoning

Limitations:
    - Context length: long tasks fill the window; requires truncation or summarization
    - Hallucination: model may fabricate tool results if not grounded
    - Brittleness: prompt format sensitivity — minor changes can break parsing
    - No learning: the model does not improve from mistakes across episodes

Fine-Tuning Recommendations:
    1. max_steps: 10–20. Beyond 20 the agent often loops or loses focus.
    2. Prompt format: few-shot examples in the system prompt strongly guide
       Thought/Action formatting. Include 2–4 gold examples per tool.
    3. Temperature: 0.0–0.3 for tool-use tasks (greedy is often best).
    4. Tool descriptions: clear, concise, with example invocations.
       Ambiguous tool names → wrong tool selection.
    5. Stopping condition: always include a Finish tool to avoid infinite loops.
    6. Error handling: return structured error messages from tools so the agent
       can diagnose and retry rather than silently failing.
"""

import ast
import operator
import re
import json
import math
import textwrap
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from dataclasses import dataclass, field
from typing import Callable

from deepdraft.core import save_figure


# ─────────────────────────────────────────────
# TOOL REGISTRY
# ─────────────────────────────────────────────

@dataclass
class Tool:
    """
    A callable tool the agent can invoke.

    Args:
        name:        unique identifier used in Action: name[args]
        description: human-readable description for the agent's system prompt
        fn:          Python callable that accepts a string argument and returns a string
    """
    name: str
    description: str
    fn: Callable[[str], str]

    def __call__(self, args: str) -> str:
        """Execute the tool and return the result as a string."""
        try:
            return self.fn(args)
        except Exception as exc:
            return f"Error: {exc}"


_BINARY_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_MATH_NAMES = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
_MATH_NAMES.update({"abs": abs, "round": round})
_MAX_EXPONENT = 1000  # guard against 10**10**10 style denial of service


def _eval_node(node: ast.AST):
    """Recursively evaluate a whitelisted arithmetic AST node."""
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
        left, right = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_EXPONENT:
            raise ValueError("exponent too large")
        return _BINARY_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.Name) and node.id in _MATH_NAMES:
        value = _MATH_NAMES[node.id]
        if callable(value):
            raise ValueError(f"'{node.id}' is a function; call it, e.g. {node.id}(2)")
        return value
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in _MATH_NAMES and callable(_MATH_NAMES[node.func.id])
            and not node.keywords):
        return _MATH_NAMES[node.func.id](*[_eval_node(a) for a in node.args])
    raise ValueError(f"unsupported expression element: {type(node).__name__}")


def _calculator(expr: str) -> str:
    """
    Safe arithmetic evaluator.

    Evaluates mathematical expressions: addition, subtraction, multiplication,
    division, powers, parentheses, constants (pi, e), and math functions
    (sqrt, log, sin, ...).

    The expression is parsed into an abstract syntax tree and only whitelisted
    node types are evaluated. This matters because an LLM chooses the input:
    Python's eval() cannot be sandboxed by removing builtins (attribute access
    such as ().__class__.__base__.__subclasses__() reaches arbitrary classes),
    so agent tools should never pass model output to eval().
    """
    try:
        tree = ast.parse(expr, mode="eval")
        return str(_eval_node(tree))
    except Exception as exc:
        return f"CalculatorError: {exc}"


# Simulated knowledge base for the mock search tool
_KNOWLEDGE_BASE = {
    "eiffel tower": "The Eiffel Tower is 330 meters tall and was built in 1889 in Paris, France.",
    "speed of light": "The speed of light in a vacuum is approximately 299,792,458 meters per second (≈ 3×10⁸ m/s).",
    "population of france": "France has a population of approximately 68 million people (2024 estimate).",
    "deepseek": "DeepSeek is a Chinese AI research company known for efficient open-weight LLMs.",
    "python": "Python is a high-level, interpreted programming language known for its readability.",
    "mamba": "Mamba is a selective state space model (SSM) that offers linear-time sequence modeling.",
}


def _search(query: str) -> str:
    """
    Simulated knowledge base lookup.

    Replaces real web search for this demo.
    In production: connect to Wikipedia API, Bing Search, or a vector store.
    """
    q = query.lower().strip()
    for key, val in _KNOWLEDGE_BASE.items():
        if key in q or all(w in q for w in key.split()):
            return val
    return f"No result found for: '{query}'. Try rephrasing or use a different tool."


_STATE: dict = {}   # shared mutable state for the lookup/store tools

def _store(args: str) -> str:
    """Store a key-value pair in working memory: 'key=value'."""
    if "=" not in args:
        return "Error: format must be 'key=value'"
    key, _, value = args.partition("=")
    _STATE[key.strip()] = value.strip()
    return f"Stored: {key.strip()} = {value.strip()}"


def _lookup(key: str) -> str:
    """Retrieve a stored value by key."""
    key = key.strip()
    if key in _STATE:
        return f"{key} = {_STATE[key]}"
    return f"Key '{key}' not found in working memory."


def build_tool_registry() -> dict[str, Tool]:
    """
    Build and return the default tool registry.

    Returns:
        dict mapping tool name → Tool instance
    """
    tools = [
        Tool(
            name="Calculator",
            description=(
                "Evaluates a math expression and returns the result. "
                "Supports +, -, *, /, **, sqrt(), log(), etc. "
                "Example: Calculator[2 ** 10] → 1024"
            ),
            fn=_calculator,
        ),
        Tool(
            name="Search",
            description=(
                "Looks up a factual query in the knowledge base. "
                "Use for facts, definitions, measurements. "
                "Example: Search[speed of light]"
            ),
            fn=_search,
        ),
        Tool(
            name="Store",
            description=(
                "Stores a value in working memory for later retrieval. "
                "Example: Store[tower_height=330] → saves 330 under 'tower_height'"
            ),
            fn=_store,
        ),
        Tool(
            name="Lookup",
            description=(
                "Retrieves a previously stored value from working memory. "
                "Example: Lookup[tower_height]"
            ),
            fn=_lookup,
        ),
    ]
    return {t.name: t for t in tools}


# ─────────────────────────────────────────────
# AGENT CORE
# ─────────────────────────────────────────────

@dataclass
class AgentStep:
    """One ReAct step: thought + action + observation."""
    thought:     str
    action_name: str
    action_args: str
    observation: str
    step_idx:    int


@dataclass
class AgentTrace:
    """Full agent execution trace for one task."""
    task:    str
    steps:   list[AgentStep] = field(default_factory=list)
    answer:  str = ""
    success: bool = False
    n_steps: int = 0


class MockLLM:
    """
    Rule-based mock LLM backend for demonstration without an API key.

    Implements a simple decision tree that mimics ReAct reasoning for a
    small set of task types: arithmetic, factual lookup, and multi-hop.

    Replace with a real LLM by implementing the same interface:
        def generate(self, prompt: str) -> str
    where the return value contains "Thought: ... Action: Name[args]" or
    "Thought: ... Action: Finish[answer]".

    Placeholder: swap this class with a call to a hosted or local LLM API for production.
    """

    def __init__(self, tools: dict[str, Tool]):
        self.tools = tools
        self._task_state: dict = {}

    def generate(self, prompt: str) -> str:
        """
        Generate the next Thought + Action given the full prompt (history included).

        Args:
            prompt: full context string up to current step

        Returns:
            string containing "Thought: ...\nAction: Name[args]"
        """
        # Extract task from prompt
        task_match = re.search(r"Task:\s*(.+?)(?:\n|$)", prompt)
        task = task_match.group(1).strip() if task_match else ""

        # Count prior steps
        n_obs = prompt.count("Observation:")

        # Multi-hop: height of Eiffel Tower in feet
        if "eiffel" in task.lower() and "feet" in task.lower():
            if n_obs == 0:
                return "Thought: I need to find the height of the Eiffel Tower first.\nAction: Search[eiffel tower]"
            elif n_obs == 1:
                return "Thought: The tower is 330 meters. I'll store this, then convert to feet.\nAction: Store[height_m=330]"
            elif n_obs == 2:
                return "Thought: 1 meter = 3.28084 feet. Let me calculate 330 * 3.28084.\nAction: Calculator[330 * 3.28084]"
            else:
                obs_lines = [l for l in prompt.split("\n") if l.startswith("Observation:")]
                last_obs = obs_lines[-1].replace("Observation:", "").strip() if obs_lines else "~1082.68"
                return f"Thought: The Eiffel Tower is approximately {last_obs} feet tall.\nAction: Finish[The Eiffel Tower is {last_obs} feet tall.]"

        # Speed of light problem
        if "speed of light" in task.lower() and ("kilometer" in task.lower() or "km" in task.lower()):
            if n_obs == 0:
                return "Thought: I need the speed of light in m/s first.\nAction: Search[speed of light]"
            elif n_obs == 1:
                return "Thought: Speed of light is 299,792,458 m/s. Convert to km/s: divide by 1000.\nAction: Calculator[299792458 / 1000]"
            else:
                obs_lines = [l for l in prompt.split("\n") if l.startswith("Observation:")]
                last_obs = obs_lines[-1].replace("Observation:", "").strip()
                return f"Thought: The speed of light is {last_obs} km/s.\nAction: Finish[The speed of light is {last_obs} km/s.]"

        # Pure math task
        if re.search(r"[\d\+\-\*/\^sqrt]", task) and "calculate" in task.lower():
            expr = re.sub(r"[^0-9\+\-\*/\.\(\)sqrt\s\^]", "", task)
            expr = expr.replace("^", "**").strip()
            if n_obs == 0:
                return f"Thought: This is an arithmetic problem. I'll use the Calculator.\nAction: Calculator[{expr}]"
            else:
                obs_lines = [l for l in prompt.split("\n") if l.startswith("Observation:")]
                last_obs = obs_lines[-1].replace("Observation:", "").strip()
                return f"Thought: The result is {last_obs}.\nAction: Finish[{last_obs}]"

        # Fallback: generic search
        if n_obs == 0:
            return f"Thought: I'll search for relevant information.\nAction: Search[{task}]"
        else:
            obs_lines = [l for l in prompt.split("\n") if l.startswith("Observation:")]
            last_obs = obs_lines[-1].replace("Observation:", "").strip() if obs_lines else "unknown"
            return f"Thought: Based on the observation, I have enough information.\nAction: Finish[{last_obs}]"


class ReActAgent:
    """
    ReAct agent loop: Thought → Action → Observation → repeat until Finish.

    Manages the prompt context (scratchpad), parses LLM outputs to extract
    actions, routes to the appropriate tool, and accumulates the trace.

    Args:
        llm:       language model backend (MockLLM or real API wrapper)
        tools:     dict of Tool instances
        max_steps: maximum iterations before forced termination
        verbose:   if True, print each step
    """

    # Regex patterns for parsing LLM output
    _THOUGHT_RE = re.compile(r"Thought:\s*(.+?)(?=\nAction:|$)", re.DOTALL)
    _ACTION_RE  = re.compile(r"Action:\s*(\w+)\[(.+?)\]", re.DOTALL)

    def __init__(
        self,
        llm,
        tools: dict[str, Tool],
        max_steps: int = 10,        # Placeholder: 10–20 for most tasks
        verbose: bool = True,
    ):
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps
        self.verbose = verbose

    def _build_system_prompt(self) -> str:
        """Construct the system prompt with tool descriptions and format instructions."""
        tool_desc = "\n".join(
            f"  {t.name}: {t.description}" for t in self.tools.values()
        )
        return textwrap.dedent(f"""
            You are a helpful assistant that solves tasks step by step.
            Use the following tools by writing exactly: Action: ToolName[argument]

            Available tools:
            {tool_desc}
              Finish: Return the final answer. Example: Finish[The answer is 42.]

            Format each step as:
            Thought: <your reasoning>
            Action: <ToolName>[<argument>]

            After receiving an Observation, continue with the next Thought.
            When you have the final answer, use Action: Finish[answer].
        """).strip()

    def _build_prompt(self, task: str, history: list[str]) -> str:
        """Assemble the full prompt from system instructions, task, and history."""
        system = self._build_system_prompt()
        hist_str = "\n".join(history)
        return f"{system}\n\nTask: {task}\n{hist_str}"

    def _parse_output(self, output: str) -> tuple[str, str, str]:
        """
        Parse LLM output into (thought, action_name, action_args).

        Returns empty strings if parsing fails — the agent will handle
        malformed output by re-prompting or terminating.
        """
        thought_match = self._THOUGHT_RE.search(output)
        action_match  = self._ACTION_RE.search(output)

        thought     = thought_match.group(1).strip() if thought_match else output.strip()
        action_name = action_match.group(1).strip() if action_match else ""
        action_args = action_match.group(2).strip() if action_match else ""

        return thought, action_name, action_args

    def run(self, task: str) -> AgentTrace:
        """
        Execute the ReAct loop for a given task.

        Args:
            task: natural language task description

        Returns:
            AgentTrace with full reasoning history and final answer
        """
        trace   = AgentTrace(task=task)
        history = []

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Task: {task}")
            print('='*60)

        for step in range(1, self.max_steps + 1):
            prompt = self._build_prompt(task, history)
            output = self.llm.generate(prompt)

            thought, action_name, action_args = self._parse_output(output)

            if not action_name:
                # Malformed output — terminate
                trace.answer = "Error: could not parse action."
                break

            if action_name == "Finish":
                trace.answer  = action_args
                trace.success = True
                observation   = f"Task complete: {action_args}"
            elif action_name in self.tools:
                observation = self.tools[action_name](action_args)
            else:
                observation = f"Unknown tool: {action_name}. Available: {list(self.tools.keys())}"

            step_record = AgentStep(
                thought=thought,
                action_name=action_name,
                action_args=action_args,
                observation=observation,
                step_idx=step,
            )
            trace.steps.append(step_record)

            # Update scratchpad
            history.append(f"Thought: {thought}")
            history.append(f"Action: {action_name}[{action_args}]")
            history.append(f"Observation: {observation}")

            if self.verbose:
                print(f"\nStep {step}")
                print(f"  Thought:     {thought}")
                print(f"  Action:      {action_name}[{action_args}]")
                print(f"  Observation: {observation}")

            if action_name == "Finish":
                break

        trace.n_steps = len(trace.steps)

        if self.verbose:
            print(f"\nAnswer: {trace.answer}")
            print(f"Steps:  {trace.n_steps} | Success: {trace.success}")

        return trace


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────

def evaluate_agent(
    agent: ReActAgent,
    tasks: list[dict],
) -> dict[str, float]:
    """
    Evaluate the agent on a list of tasks with ground-truth answers.

    Args:
        agent: ReActAgent instance
        tasks: list of dicts with keys 'task' and 'expected_answer'

    Returns:
        dict with success_rate, avg_steps, and per-task results
    """
    results = []
    for item in tasks:
        trace = agent.run(item["task"])
        expected = str(item.get("expected_answer", "")).lower().strip()
        actual   = trace.answer.lower().strip()
        # Loose match: check if expected appears anywhere in the answer
        match = expected in actual or actual in expected
        results.append({
            "task":    item["task"],
            "success": trace.success and match,
            "steps":   trace.n_steps,
            "answer":  trace.answer,
        })

    success_rate = sum(r["success"] for r in results) / len(results)
    avg_steps    = sum(r["steps"]   for r in results) / len(results)
    return {
        "success_rate": success_rate,
        "avg_steps":    avg_steps,
        "per_task":     results,
    }


# ─────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────

def plot_trace(trace: AgentTrace, filename: str = "react_trace.png") -> None:
    """
    Visualize the agent's ReAct trace as a vertical timeline.

    Each row shows one step: step index, tool name, a truncated thought,
    and the observation. Color-coded by tool type.
    """
    n_steps = len(trace.steps)
    if n_steps == 0:
        print("No steps to plot.")
        return

    tool_colors = {
        "Calculator": "#4CAF50",
        "Search":     "#2196F3",
        "Store":      "#FF9800",
        "Lookup":     "#9C27B0",
        "Finish":     "#E91E63",
    }

    fig, ax = plt.subplots(figsize=(12, max(4, n_steps * 1.2)))

    for i, step in enumerate(trace.steps):
        y = n_steps - i - 0.5
        color = tool_colors.get(step.action_name, "#607D8B")

        # Step box
        rect = mpatches.FancyBboxPatch(
            (0.0, y - 0.4), 11.8, 0.8,
            boxstyle="round,pad=0.05",
            facecolor=color, alpha=0.15, edgecolor=color, linewidth=1.5,
        )
        ax.add_patch(rect)

        # Step number
        ax.text(0.15, y, f"#{step.step_idx}", va="center", ha="center",
                fontsize=9, fontweight="bold", color=color)

        # Tool name badge
        ax.text(0.7, y, step.action_name, va="center", ha="center",
                fontsize=8, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.8, edgecolor="none"),
                color="white")

        # Thought (truncated)
        thought_short = (step.thought[:60] + "…") if len(step.thought) > 60 else step.thought
        ax.text(1.5, y + 0.15, f"Thought: {thought_short}",
                va="center", fontsize=7.5, color="#333333")

        # Observation (truncated)
        obs_short = (step.observation[:70] + "…") if len(step.observation) > 70 else step.observation
        ax.text(1.5, y - 0.15, f"Obs: {obs_short}",
                va="center", fontsize=7, color="#555555", style="italic")

    ax.set_xlim(0, 12)
    ax.set_ylim(-0.5, n_steps + 0.3)
    ax.axis("off")
    ax.set_title(
        f"ReAct Trace — Task: {trace.task[:80]}\nAnswer: {trace.answer[:80]}",
        fontsize=10, pad=10,
    )
    plt.tight_layout()
    save_figure(filename, dpi=150)


def plot_evaluation_summary(eval_results: dict) -> None:
    """
    Bar chart of success rate and average steps across evaluated tasks.
    """
    per_task = eval_results["per_task"]
    labels  = [f"Task {i+1}" for i in range(len(per_task))]
    success = [int(r["success"]) for r in per_task]
    steps   = [r["steps"] for r in per_task]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].bar(labels, success, color=["green" if s else "red" for s in success])
    axes[0].set_yticks([0, 1]); axes[0].set_yticklabels(["Fail", "Success"])
    axes[0].set_title(f"Task Success (rate={eval_results['success_rate']:.0%})")
    axes[0].tick_params(axis="x", rotation=30)

    axes[1].bar(labels, steps, color="steelblue")
    axes[1].axhline(eval_results["avg_steps"], color="darkorange", linestyle="--",
                    label=f"avg={eval_results['avg_steps']:.1f}")
    axes[1].set_title("Steps per Task"); axes[1].legend()
    axes[1].tick_params(axis="x", rotation=30)

    plt.suptitle("ReAct Agent — Evaluation Summary")
    plt.tight_layout()
    save_figure("react_evaluation.png", dpi=150)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=== ReAct Agent — Reasoning + Acting ===")

    # 1. Build tools and agent
    tools = build_tool_registry()
    llm   = MockLLM(tools)
    agent = ReActAgent(llm, tools, max_steps=8, verbose=True)

    # 2. Demo tasks
    demo_tasks = [
        {
            "task": "What is the height of the Eiffel Tower in feet?",
            "expected_answer": "1082",
        },
        {
            "task": "What is the speed of light in kilometers per second?",
            "expected_answer": "299792",
        },
    ]

    print("\n" + "="*60)
    print("Running demo tasks...")
    traces = []
    for item in demo_tasks:
        trace = agent.run(item["task"])
        traces.append(trace)

    # 3. Visualize the first trace
    plot_trace(traces[0], filename="react_trace.png")

    # 4. Batch evaluation
    print("\n--- Batch Evaluation ---")
    eval_results = evaluate_agent(agent, demo_tasks)
    print(f"Success rate: {eval_results['success_rate']:.0%}")
    print(f"Avg steps:    {eval_results['avg_steps']:.1f}")
    plot_evaluation_summary(eval_results)


if __name__ == "__main__":
    main()
