from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from init_llm import deepseek_llm


# 1. 定义结构
class Movie(BaseModel):
    title: str = Field(description="电影标题")
    year: int = Field(description="上映年份")

# 2. 设置提示词
prompt = ChatPromptTemplate.from_template("""
回答用户问题。
问题：{question}
你必须始终输出一个包含title(电影标题)和year(上映年份)的 JSON 对象。
""")

# 3. 创建链
chain = prompt | deepseek_llm | JsonOutputParser(pydantic_object=Movie)

# 4. 调用（返回字典）
response = chain.invoke({"question": "介绍电影《盗梦空间》"})
print(response)