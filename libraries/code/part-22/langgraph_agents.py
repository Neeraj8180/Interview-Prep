# LangGraph stateful agents — Part 22 code examples
# langgraph 0.3.x | Python 3.11+
# Install: pip install langgraph langchain-core

from typing import Annotated, List
from typing_extensions import TypedDict
import operator


# ── 1. Multi-turn chatbot with LangGraph + memory ─────────────────────────────

def demo_multiturn_chatbot():
    print("=" * 55)
    print("1. Multi-Turn Chatbot with LangGraph")
    print("=" * 55)

    try:
        from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
        from langgraph.graph import StateGraph, END
        from langgraph.checkpoint.memory import MemorySaver

        class ChatState(TypedDict):
            messages: Annotated[List[BaseMessage], operator.add]

        def chat_node(state: ChatState) -> dict:
            """Call LLM with current conversation history."""
            try:
                from langchain_openai import ChatOpenAI
                llm    = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
                system = SystemMessage("You are a helpful, concise assistant.")
                resp   = llm.invoke([system] + state["messages"])
                return {"messages": [resp]}
            except Exception:
                # Fallback: echo last message for demo
                last = state["messages"][-1].content
                return {"messages": [AIMessage(f"(Demo echo) You said: '{last}'")]}

        # Build graph
        workflow = StateGraph(ChatState)
        workflow.add_node("chat", chat_node)
        workflow.set_entry_point("chat")
        workflow.add_edge("chat", END)

        app = workflow.compile(checkpointer=MemorySaver())
        cfg = {"configurable": {"thread_id": "demo-session"}}

        # Multi-turn conversation
        turns = [
            "Hi! My name is Alice and I'm a Python developer.",
            "What programming language do I prefer?",   # tests memory
        ]
        for user_msg in turns:
            result = app.invoke(
                {"messages": [HumanMessage(user_msg)]},
                config=cfg,
            )
            print(f"\nUser: {user_msg}")
            print(f"AI:   {result['messages'][-1].content[:100]}")

        # Show accumulated state
        state = app.get_state(cfg)
        print(f"\nTotal messages in state: {len(state.values['messages'])}")

    except ImportError as e:
        print(f"LangGraph not installed: {e}")
        print("Install: pip install langgraph langchain-core")
        _show_langgraph_concept()


def _show_langgraph_concept():
    print("\nLangGraph pattern:")
    print("""
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated, List
import operator

class State(TypedDict):
    messages: Annotated[List, operator.add]  # append-only

def my_node(state: State) -> dict:
    # Process state, return dict with updates
    new_msg = call_llm(state["messages"])
    return {"messages": [new_msg]}

workflow = StateGraph(State)
workflow.add_node("llm", my_node)
workflow.set_entry_point("llm")
workflow.add_edge("llm", END)
app = workflow.compile(checkpointer=MemorySaver())

result = app.invoke(
    {"messages": [HumanMessage("Hello")]},
    config={"configurable": {"thread_id": "session-1"}},
)
""")


# ── 2. Conditional routing graph ──────────────────────────────────────────────

def demo_conditional_routing():
    print("\n" + "=" * 55)
    print("2. Conditional Routing (ReAct-style loop)")
    print("=" * 55)

    try:
        from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
        from langgraph.graph import StateGraph, END

        class AgentState(TypedDict):
            messages: Annotated[List, operator.add]
            step_count: int

        def agent_node(state: AgentState) -> dict:
            """Decide: call tool or finish."""
            step = state.get("step_count", 0)
            last = state["messages"][-1].content if state["messages"] else ""

            # Simulate: first step considers using a tool, second step finishes
            if step == 0 and "weather" in last.lower():
                ai_msg = AIMessage(
                    content="I need to check the weather.",
                    tool_calls=[{
                        "id": "tc_001",
                        "name": "get_weather",
                        "args": {"city": "Paris"},
                    }]
                )
            else:
                ai_msg = AIMessage(content=f"Final answer after {step} steps.")
            return {"messages": [ai_msg], "step_count": step + 1}

        def tool_node(state: AgentState) -> dict:
            """Execute tools called by the agent."""
            last = state["messages"][-1]
            results = []
            for tc in getattr(last, "tool_calls", []):
                if tc["name"] == "get_weather":
                    result = f"Weather in {tc['args']['city']}: 22°C, sunny"
                else:
                    result = f"Tool {tc['name']} not found"
                results.append(ToolMessage(content=result, tool_call_id=tc["id"]))
            return {"messages": results}

        def should_continue(state: AgentState) -> str:
            """Route: call tool or end."""
            last = state["messages"][-1]
            if hasattr(last, "tool_calls") and last.tool_calls:
                return "tools"
            return END

        workflow = StateGraph(AgentState)
        workflow.add_node("agent", agent_node)
        workflow.add_node("tools", tool_node)
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges("agent", should_continue, {
            "tools": "tools",
            END:     END,
        })
        workflow.add_edge("tools", "agent")  # loop back

        app = workflow.compile()

        result = app.invoke({
            "messages": [HumanMessage("What is the weather in Paris?")],
            "step_count": 0,
        })

        print("Conversation trace:")
        for msg in result["messages"]:
            msg_type = type(msg).__name__
            content  = msg.content[:80] if msg.content else "(tool call)"
            print(f"  [{msg_type}]: {content}")

    except ImportError as e:
        print(f"LangGraph not installed: {e}")


# ── 3. State persistence demonstration ───────────────────────────────────────

def demo_state_inspection():
    print("\n" + "=" * 55)
    print("3. State Inspection (TypedDict + reducers)")
    print("=" * 55)

    try:
        from langchain_core.messages import HumanMessage, AIMessage
        from langgraph.graph import StateGraph, END
        from langgraph.checkpoint.memory import MemorySaver

        class CountingState(TypedDict):
            messages:  Annotated[List, operator.add]  # append
            visit_count: int                           # replace (last write wins)

        def count_node(state: CountingState) -> dict:
            n = state.get("visit_count", 0) + 1
            return {
                "messages":    [AIMessage(f"Visit #{n}")],
                "visit_count": n,
            }

        workflow = StateGraph(CountingState)
        workflow.add_node("count", count_node)
        workflow.set_entry_point("count")
        workflow.add_edge("count", END)
        app = workflow.compile(checkpointer=MemorySaver())
        cfg = {"configurable": {"thread_id": "counter-thread"}}

        for _ in range(3):
            result = app.invoke(
                {"messages": [HumanMessage("hi")], "visit_count": 0},
                config=cfg,
            )

        state = app.get_state(cfg)
        print(f"visit_count (replace reducer): {state.values['visit_count']}")
        print(f"messages (append reducer):    {len(state.values['messages'])} total")
        print("\nKey insight:")
        print("  messages: Annotated[List, operator.add] → APPENDS (history grows)")
        print("  visit_count: int                        → REPLACES (last write wins)")

    except ImportError as e:
        print(f"LangGraph not installed: {e}")


if __name__ == "__main__":
    demo_multiturn_chatbot()
    demo_conditional_routing()
    demo_state_inspection()
