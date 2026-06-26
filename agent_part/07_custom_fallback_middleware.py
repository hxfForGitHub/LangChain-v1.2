"""自定义 ModelFallbackMiddleware，显式调用并打印执行流程"""

from langchain.agents import create_agent
from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool

from env_utils import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, OPENAI_API_KEY, OPENAI_BASE_URL

# ─── 准备模型 ─────────────────────────────────────────────
good_model = init_chat_model(
    model="deepseek-chat",
    model_provider="deepseek",
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
)

bad_model = init_chat_model(
    model="non-existent-model",
    model_provider="deepseek",
    api_key="invalid-key",
    base_url="https://api.invalid.url",
)

# ─── 准备工具 ─────────────────────────────────────────────
@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气"""
    return f"{city}的天气为晴朗，25°C。"


# ═══════════════════════════════════════════════════════════
# 自定义 ModelFallbackMiddleware - 显式实现 wrap_model_call
# ═══════════════════════════════════════════════════════════
class MyFallbackMiddleware(AgentMiddleware):
    """手动实现的模型回退中间件，每一步都打印"""

    def __init__(self, *fallback_models):
        super().__init__()
        self.models = list(fallback_models)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler,
    ) -> ModelResponse:
        """
        handler 是什么？
        - 它是一个可调用对象，指向 Agent 内部的"真正调用模型"的函数
        - 调用 handler(request) = 让 Agent 去执行 LLM.invoke()
        - 调用 handler(request.override(model=xxx)) = 换模型后再让 Agent 调用
        """
        print("\n>>> [中间件] wrap_model_call 开始执行")

        # 第一步：尝试主模型（create_agent 时指定的 model）
        print(">>> [中间件] 第1步: 调 handler(request) → 主模型开始调用...")
        try:
            result = handler(request)
            print(">>> [中间件] 主模型调用成功 ✅")
            return result
        except Exception as e:
            print(f">>> [中间件] 主模型调用失败 ❌: {type(e).__name__}")

        # 第二步：逐个尝试备用模型
        for i, fb_model in enumerate(self.models, 1):
            print(f">>> [中间件] 第{i+1}步: 调 handler(request.override(model=fb_model)) → 备用模型{i}开始调用...")
            try:
                result = handler(request.override(model=fb_model))
                print(f">>> [中间件] 备用模型{i}调用成功 ✅")
                return result
            except Exception as e:
                print(f">>> [中间件] 备用模型{i}调用失败 ❌: {type(e).__name__}")

        print(">>> [中间件] 所有模型都失败了 💥")
        raise Exception("所有模型都不可用")

    # 也可以顺便看看 before/after 会不会被触发
    def before_model(self, state):
        print(">>> [钩子] before_model 被触发了")
        return state

    def after_model(self, state):
        print(">>> [钩子] after_model 被触发了")
        return state

    def wrap_tool_call(self, request, handler):
        print(f">>> [中间件] wrap_tool_call 开始执行 → 调工具: {request.tool.name}")
        result = handler(request)
        print(f">>> [中间件] 工具返回: {result}")
        return result


# ═══════════════════════════════════════════════════════════
# 使用自定义中间件
# ═══════════════════════════════════════════════════════════
print("=" * 60)
print("自定义 ModelFallbackMiddleware - 显式调用流程")
print("=" * 60)

my_mw = MyFallbackMiddleware(good_model)   # 主模型失败时，回退到 good_model

agent = create_agent(
    model=bad_model,           # 主模型 = 故意失败的模型
    tools=[get_weather],
    middleware=[my_mw],
)

print("\n>>> 开始 agent.invoke()...\n")

resp = agent.invoke({
    "messages": [{"role": "user", "content": "北京的天气怎么样？"}]
})

print("\n>>> agent.invoke() 返回")
content = resp["messages"][-1].content
print(f"最终回复: {content[:80]}...")