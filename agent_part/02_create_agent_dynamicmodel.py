
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelResponse, ModelRequest
from langchain.chat_models import init_chat_model
from langchain_core.messages import function
from langchain_core.tools import tool

from env_utils import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL

@tool
def get_current_location() -> str:
    """获取当前位置。"""
    return "当前位置为北京市。"

@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气信息。"""
    return f"{city}的天气为晴朗，25°C。"


basic_model = init_chat_model(
    model="deepseek-chat",
    model_provider="deepseek",
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
)


advanced_model = init_chat_model(
    model="qwen-plus",
    model_provider="openai",
    api_key=DASHSCOPE_API_KEY,
    base_url=DASHSCOPE_BASE_URL,

)


# 还有一个中间件叫 @wrap_tool_call, 参数也是（request, handler）但两个request的类型是不同的。
@wrap_model_call
def dynamic_model_selection(request:ModelRequest, handler) -> ModelResponse:
    print("request : ",request)
    # 判断消息条数，如果小于3条使用 basic_model，否则使用advanced_model
    message_count = len(request.state['messages'])
    if message_count < 3:
        model = basic_model
    else:
        model = advanced_model

    # 这里用到了 request.override，可以改变一个或多个参数，详情看源码
    return handler(request.override(model=model))



agent = create_agent(
    model=basic_model,
    tools=[get_current_location, get_weather],
    # middleware 只接受 AgentMiddleware 实例， 加了 @wrap_model_call 装饰器的方法，会把这个方法变成一个 AgentMiddleware 实例
    # middleware 只收 AgentMiddleware 实例，不收裸函数，也不收类
    # @wrap_model_call 做的事就是：把你的函数打包成一个 AgentMiddleware 的匿名子类实例，让它能塞进 middleware 列表里
    middleware=[dynamic_model_selection],
)

response = agent.invoke({"messages":[
    {"role":"system","content":"你是一个天气助手"},
    {"role":"user","content":"我现在在的位置天气如何？"}
]})
print(response)


