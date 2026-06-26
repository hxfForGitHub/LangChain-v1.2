"模型调用工具"
from pprint import pprint

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from init_llm import deepseek_llm


# 1.创建工具
@tool
def get_weather(local: str) -> str:
    """获取天气信息"""
    return f"{local}天气非常晴朗！"

# 2.给模型绑定工具
model_with_tools = deepseek_llm.bind_tools([get_weather])

# 3.准备messages
messages = []
humanMessage = HumanMessage(content="北京天气是什么？")
messages.append(humanMessage)


# 4.模型不会真正执行调用工具，只是知道要调用工具
response = model_with_tools.invoke(messages)

print(repr(response))

messages.append(response)

# 一个小总结：
# LLM 如果不是 agent 的话，LLM 绑定了 tools，去 invoke ,
# 但它并不会真正去调用 tools, 它只是会说，我要去调用，但不会真正去用。
# 这个时候，是需要手动去找到LLM决定要调用的工具，然后手动调用工具。
# 再把得到的结果，添加到 messages 列表中 中。
# 并且： get_weather.invoke(tool_call) 这样是返回一个ToolMessage 的，
# 并不是 get_weather.invoke，参数不同，返回的类型是不同的，如果只是传入字符串参数，它返回也是一个字符串
# 但传入的是一个tool_call完整字典，langchain 会根据这个字典，调用对应的工具，并将结果封装成一个ToolMessage，
# 然后再添加到 messages 列表中 中。



# 5.获取工具调用信息
if response.tool_calls:
    for tool_call in response.tool_calls:
        # 打印工具调用信息
        if tool_call["name"] == "get_weather":
            # 手动调用工具
            tool_result = get_weather.invoke(tool_call)

            print(type(tool_result))
            # print(tool_result)
            print(repr(tool_result))

            messages.append(tool_result)

# 6.模型会根据工具调用结果，生成最终回复
# print("messages:", messages)
final_response = model_with_tools.invoke(messages)
# print(type(final_response))
# print(final_response)
print(repr(final_response))




