from openai import OpenAI
import json

client = OpenAI(
    base_url="http://146.190.191.13/v1",
    api_key="changeme",
)

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City and country"}
                },
                "required": ["location"],
            },
        },
    }
]

stream = client.chat.completions.create(
    model="Qwen/Qwen3-14B-AWQ",
    messages=[
        {"role": "system", "content": "/no_think"},
        {"role": "user", "content": "What's the weather in New York?"},
    ],
    tools=tools,
    tool_choice="auto",
    max_tokens=256,
    stream=True,
)

tool_calls = {}
for chunk in stream:
    delta = chunk.choices[0].delta

    if delta.content:
        print(delta.content, end="", flush=True)

    if delta.tool_calls:
        for tc in delta.tool_calls:
            idx = tc.index
            if idx not in tool_calls:
                tool_calls[idx] = {"id": tc.id, "name": tc.function.name, "arguments": ""}
                print(f"\n>> Calling tool: {tc.function.name}(", end="", flush=True)
            if tc.function.arguments:
                tool_calls[idx]["arguments"] += tc.function.arguments
                print(tc.function.arguments, end="", flush=True)

    if chunk.choices[0].finish_reason == "tool_calls":
        print(")")

print()

if tool_calls:
    for idx, tc in tool_calls.items():
        print(f"Tool call: {tc['name']}({tc['arguments']})")

        result = json.dumps({"temperature": "45°F", "condition": "Cloudy", "location": "New York"})

        messages = [
            {"role": "system", "content": "/no_think"},
            {"role": "user", "content": "What's the weather in New York?"},
            {"role": "assistant", "tool_calls": [{"id": tc["id"], "type": "function", "function": {"name": tc["name"], "arguments": tc["arguments"]}}]},
            {"role": "tool", "tool_call_id": tc["id"], "content": result},
        ]

        follow_up = client.chat.completions.create(
            model="Qwen/Qwen3-14B-AWQ",
            messages=messages,
            max_tokens=256,
            stream=True,
        )

        print("Assistant: ", end="")
        for chunk in follow_up:
            if chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="", flush=True)
        print()
else:
    print("No tool calls made.")
