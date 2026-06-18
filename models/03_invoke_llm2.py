from init_llm import deepseek_llm

# 这个 batch 是需要三个问题都准备好了，才一次性返回
resp = deepseek_llm.batch([
    "请介绍一下你自己",
    "飞机为什么会飞",
    "什么是大模型",
])
for item in resp:
    print(item)


# 哪个已经有答案，就先返回哪个。不会等全部都有结果了再返回。
resp = deepseek_llm.batch_as_completed([
    "请介绍一下你自己",
    "飞机为什么会飞",
    "什么是大模型",
])

for item in resp:
    print(item)
