from typing import TypedDict, Annotated

from init_llm import deepseek_llm
# 使用TypedDict定义嵌套结构化输出模型
# 定义嵌套结构体

# 这只是返回格式的另外一种写法，知道就行，用 Pydantic 模型更方便， 用好一种就可以了。

class Actor(TypedDict):
    name: Annotated[str, "演员姓名"]
    role: Annotated[str, "演员在电影中的角色"]

class Movie(TypedDict):
    title: Annotated[str, "电影标题"]
    year: Annotated[int, "电影上映年份"]
    director: Annotated[str, "电影导演"]
    rating: Annotated[float, "电影评分"]
    cast: Annotated[list[Actor], "电影演员列表"]

# 绑定结构化输出模型
model_with_structured_output = deepseek_llm.with_structured_output(Movie)

# resp = model_with_structured_output.invoke("介绍下电影《泰坦尼克号》")
# resp = model_with_structured_output.invoke("介绍下电影《78》不超过10字,禁止返回电影年份和导演任何信息")


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



# 使用TypedDict定义简单结构化输出模型
# class Movie(TypedDict):
#     title: Annotated[str, "电影标题"]
#     year: Annotated[int, "电影上映年份"]
#     director: Annotated[str, "电影导演"]
#     rating: Annotated[float, "电影评分"]
#
# # 绑定结构化输出模型
# model_with_structured_output = deepseek_llm.with_structured_output(Movie)
#
# resp = model_with_structured_output.invoke("介绍下电影《泰坦尼克号》")
#
# print(type(resp))
# print(resp)




