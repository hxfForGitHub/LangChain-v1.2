"pydantic 模型返回结构化数据"
from pydantic import BaseModel, Field

from init_llm import deepseek_llm

# 返回嵌套对象
class Actor(BaseModel):
    name: str = Field(description="演员姓名")
    role: str = Field(description="演员在电影中的角色")


class Movie(BaseModel):
    title: str = Field(description="电影标题")
    year: int = Field(description="电影上映年份")
    director: str = Field(description="电影导演")
    rating: float = Field(description="电影评分")
    cast: list[Actor] = Field(description="电影演员列表")


model_with_structured_output = deepseek_llm.with_structured_output(Movie,include_raw=True)
# resp = model_with_structured_output.invoke("介绍下电影《泰坦尼克号》")


# 简单地使用 "介绍下电影《泰坦尼克号》" ， 它只返回了 title， 导致报错，无法生成正确的 Movie 对象，
# 把提示词写得漂亮一些，结果就对了。
# with_structured_output 也是一种tool call， tool_name 就是类名 Movie
prompt = """
请从以下内容中提取电影信息，严格按照指定的JSON格式输出：
- 电影标题 (title): 泰坦尼克号
- 上映年份 (year): 
- 导演 (director): 
- 评分 (rating): 
- 演员列表 (cast): 每个演员包含 name 和 role

请提取电影《泰坦尼克号》的完整信息。
"""
resp = model_with_structured_output.invoke(prompt)


print(type(resp))
print(resp)


# 定义一个 Pydantic 模型，用于结构化输出简单对象
# class Movie(BaseModel):
#     title: str = Field(description="电影标题")
#     year: int = Field(description="电影上映年份")
#     director: str = Field(description="电影导演")
#     rating: float = Field(description="电影评分")
#
#
# model_with_structured_output = deepseek_llm.with_structured_output(Movie,)
#
# resp = model_with_structured_output.invoke("介绍下电影《78》不超过10字,禁止返回电影年份和导演任何信息")
#
# print(type(resp))
# print(resp)

