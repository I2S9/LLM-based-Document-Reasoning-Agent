# Example: Using the Planner

from src.agent.planner import plan

# Test different queries
queries = [
    "What is machine learning?",
    "Explain neural networks",
    "How does natural language processing work?"
]

print("Planner Examples")
print("=" * 60)

for query in queries:
    steps = plan(query)
    print(f"\nQuery: '{query}'")
    print(f"Steps ({len(steps)}):")
    for i, step in enumerate(steps, 1):
        print(f"  {i}. {step}")

