"""演示 LangChain 预置中间件的使用

预置中间件类：
1. SummarizationMiddleware   - 对话历史过长时自动总结
2. HumanInTheLoopMiddleware  - 工具调用前需要人工确认
3. ModelFallbackMiddleware   - 模型失败时自动切换备用模型

用法：传给 create_agent 的 middleware 参数
"""

from langchain.agents import create_agent
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    ModelFallbackMiddleware,
    SummarizationMiddleware,
)
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool

from env_utils import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
)

# ─── 准备模型 ─────────────────────────────────────────────
main_model = init_chat_model(
    model="deepseek-chat",
    model_provider="deepseek",
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
)

fallback_model = init_chat_model(
    model="gpt-4o-mini",
    model_provider="openai",
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
)

# ─── 准备工具 ─────────────────────────────────────────────
@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气"""
    return f"{city}的天气为晴朗，25°C。"


@tool
def calculate(expression: str) -> str:
    """计算数学表达式"""
    try:
        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误: {e}"


# ═══════════════════════════════════════════════════════════
# Demo 1: SummarizationMiddleware
# ═══════════════════════════════════════════════════════════
def demo_summarization():
    """自动总结中间件：消息超过一定数量时自动压缩历史"""
    print("=" * 60)
    print("Demo 1: SummarizationMiddleware（自动总结历史）")
    print("=" * 60)

    summarization_mw = SummarizationMiddleware(
        model=main_model,
        # trigger=("messages", 6),   # 消息达到 6 条时触发总结
        trigger=("fraction", 0.8),    # 或者：token 使用量达到 80% 时触发
        keep=("messages", 4),         # 总结后保留最近 4 条消息
    )

    agent = create_agent(
        model=main_model,
        tools=[get_weather],
        middleware=[summarization_mw],
    )

    # 连续发多条消息，触发自动总结
    conversations = [
        "北京的天气怎么样？",
        "那上海呢？",
        "广州呢？",
        "深圳呢？",
        "杭州呢？",
        "成都呢？",     # 这里可能触发总结
        "再来一个：南京呢？",
    ]

    for msg in conversations:
        print(f"\n用户: {msg}")
        resp = agent.invoke({"messages": [{"role": "user", "content": msg}]})
        content = resp["messages"][-1].content
        print(f"助手: {content[:80]}...")


# ═══════════════════════════════════════════════════════════
# Demo 2: HumanInTheLoopMiddleware
# ═══════════════════════════════════════════════════════════
def demo_human_in_the_loop():
    """人工确认中间件：指定工具调用前需要用户批准"""
    print("\n" + "=" * 60)
    print("Demo 2: HumanInTheLoopMiddleware（人工确认）")
    print("=" * 60)
    print("注意：此中间件会暂停等待输入，需要手动确认")
    print("仅在交互式终端中有效，脚本模式会报错\n")

    human_mw = HumanInTheLoopMiddleware(
        interrupt_on={
            # key=工具名, value=True 表示需要确认
            "get_weather": True,
            "calculate": True,
        },
    )

    agent = create_agent(
        model=main_model,
        tools=[get_weather, calculate],
        middleware=[human_mw],
    )

    resp = agent.invoke({
        "messages": [{"role": "user", "content": "北京的天气怎么样？"}]
    })
    content = resp["messages"][-1].content
    print(f"最终回复: {content}")


# ═══════════════════════════════════════════════════════════
# Demo 3: ModelFallbackMiddleware
# ═══════════════════════════════════════════════════════════
def demo_model_fallback():
    """模型回退中间件：主模型失败时自动切换备用模型"""
    print("\n" + "=" * 60)
    print("Demo 3: ModelFallbackMiddleware（模型回退）")
    print("=" * 60)

    # 制造一个"不可用"的模型，触发回退
    bad_model = init_chat_model(
        model="non-existent-model",     # 不存在的模型，故意让它失败
        model_provider="deepseek",
        api_key="invalid-key",
        base_url="https://api.invalid.url",
    )

    fallback_mw = ModelFallbackMiddleware(
        bad_model,          # 先用这个（会失败）
        main_model,         # 失败了切换到主模型
        fallback_model,   # 还可以再加更多备用模型
    )

    agent = create_agent(
        model=bad_model,        # Agent 默认用 bad_model
        tools=[get_weather],
        middleware=[fallback_mw],
    )

    print(">>> 主模型故意设置为不存在的模型，会自动回退到备用模型...\n")
    print(">>> 执行流程追踪：")
    print(">>> 1. agent.invoke() 开始")
    print(">>> 2.   → Agent 内部循环启动")
    print(">>> 3.     → wrap_model_call = ModelFallbackMiddleware 开始执行")
    print(">>> 4.       中间件内部：尝试 handler(request) 即 bad_model.invoke()（会失败）")
    print(">>> 5.       中间件内部：捕获异常")
    print(">>> 6.       中间件内部：切换到 main_model，再调 handler(request.override(model=main_model))")
    print(">>> 7.       中间件内部：main_model 调用成功，返回结果")
    print(">>> 8.     → wrap_model_call 结束")
    print(">>> 9.   → Agent 判断需要调工具")
    print(">>> 10.    → wrap_tool_call（调 get_weather）")
    print(">>> 11.    → 工具返回结果")
    print(">>> 12.  → Agent 再次调用 LLM 生成最终回复")
    print(">>> 13.  → agent.invoke() 返回\n")

    resp = agent.invoke({
        "messages": [{"role": "user", "content": "北京的天气怎么样？"}]
    })
    content = resp["messages"][-1].content
    print(f"最终回复: {content[:80]}...")


# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    # 逐个运行演示（HumanInTheLoop 需要交互式输入，单独运行）

    # demo_summarization()

    # 取消注释来运行人工确认演示（需要终端交互）
    # demo_human_in_the_loop()

    demo_model_fallback()