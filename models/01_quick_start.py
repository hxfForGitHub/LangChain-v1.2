"""
创建一个Agent，调用工具回答用户问题，
"""
import sys as _sys
from pathlib import Path

# _sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain.agents import create_agent
from langchain_core.tools import tool

from init_llm import deepseek_llm


# from my_llm import deepseek_llm

@tool
def get_weather(location: str) -> str:
    """
    获取指定位置的天气信息
    """
    return f"天气信息：{location}的天气是晴朗的"


agent = create_agent(
    model=deepseek_llm,
    tools=[get_weather],
    system_prompt="你是一个天气助手，可以帮助用户获取指定位置的天气信息"
)

resp = agent.invoke({"messages": [{"role": "user", "content": "广州"}]})

print(type(resp))
print(resp)
