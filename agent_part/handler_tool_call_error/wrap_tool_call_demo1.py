"""
 优雅处理工具调用错误
"""
import requests
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt.tool_node import ToolCallRequest

from init_llm import deepseek_llm


@tool
def get_stock_price(symbol: str) -> str:
    """
    获取指定股票代码的当前价格
    Arg:
        symbol: 股票代码（例如：TCEHY）
    Returns:
        股票当前价格（例如："股票 TCEHY 当前价格为 123.45"）
    """
    print(f"=====调用股票查询工具: {symbol}")
    try:
        # 模拟可能失败的API调用
        response = requests.get(f"https://api.xxx.com/stocks/{symbol}", timeout=1)
        return f"股票 {symbol} 当前价格为 {response['price']}"
    except requests.exceptions.RequestException as e:
        print(f"=====查询股票数据失败: {str(e)}")
        raise Exception(f"查询股票数据失败: {str(e)}")


@wrap_tool_call
def handle_tool_call_error(request:ToolCallRequest ,handler):
    print("request",request)

    # 介入错误处理
    try:
        result = handler(request)
        print("tool call finish")
        return result
    except Exception as e:
        return ToolMessage(
            content=f"当前股票查询服务不可用，错误信息: {str(e)}",
            tool_call_id=request.tool_call["id"]
        )




agent = create_agent(
    model=deepseek_llm,

    # 对于tools & middleware 的关系有如下两种
    # 1、多个tools 一个 middleware, 每个 tool 执行的时候都会受 middleware的影响，如果不想受到影响，只能在 middleware 这个 @wrap_tool_call 方法里面进行针对性处理或过滤
    # 2、如果有多个 middleware, 就会按链式，包洋葱似的，一层套一层地执行。
        # ├─ Agent 判断要调 toolA
        # │  ├─ middleware1: "A 开始"
        # │  │  ├─ middleware2: "B 开始"
        # │  │  │  ├─ toolA 执行...
        # │  │  │  └─ toolA 返回
        # │  │  └─ middleware2: "B 结束"
        # │  └─ middleware1: "A 结束"
        # │
        # ├─ Agent 判断还要调 toolB
        # │  ├─ middleware1: "A 开始"
        # │  │  ├─ middleware2: "B 开始"
        # │  │  │  ├─ toolB 执行...
        # │  │  │  └─ toolB 返回
        # │  │  └─ middleware2: "B 结束"
        # │  └─ middleware1: "A 结束"
        # │
        # └─ Agent 判断不需要调工具了，结束

    tools=[get_stock_price],
    middleware=[handle_tool_call_error]
)

result = agent.invoke({"messages":[{"role":"user","content":"查询TCEHY的股票价格"}]})

print("result",result)

print(result["messages"][-1].content)



