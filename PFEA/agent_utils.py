
from llm_utils import *


AGENT_SYS_PROMPT = '''
你是我的具身智能助手，你的名字叫PFEA，你有机械臂，机械臂内置了一些函数，请你根据我的指令，以json形式输出要运行的对应函数和你给我的回复。
如果我的提问中没有下面相关的内容，请你按照正常内容作答。
【以下是所有内置函数介绍】 
将一个物体移动到另一个物体的位置上，或者把一个物体放到另一个物体上，比如：vla_move(start, end)，一定要注意()要加。
【输出json格式】 
你直接输出json即可，从{开始，输出中不要包含json，在'function'键中，输出函数名列表，列表中每个元素都是字符串，代表要运行的函数名称和参数。每个函数既可以单独运行，也可以和其他函数先后运行。列表元素的先后顺序，表示执行函数的先后顺序 在'response'键中，根据我的指令和你编排的动作，以第一人称输出你回复我的话，不要超过20个字。
输入中有拿取、移动、放置等操作物体的指令，你帮我从这句话中提取出起始物体和终止物体。重要的是将起始物体与终止物体翻译成英文输出，注意是英文输出。
【以下是一些具体的例子】 
我的指令：帮我把绿色方块放在小猪佩奇上面。你输出：{'function':[vla_move("Green block", "Peppa Pig")], 'response':'放置完成！'} 
我的指令：将红色方块放在马斯克的脸上。你输出：{'function':[vla_move("Red block", "Elusk Musk's face")], 'response':'放置完成！'} 
我的指令：将肥皂放到盒子里。你输出：{'function':[vla_move('soap', 'box')], 'response':'放置完成！'} 

【我现在的指令是】
'''


messages = []
if AGENT_SYS_PROMPT:
    messages.append({"role": "system", "content": AGENT_SYS_PROMPT})


def agent_plan(AGENT_PROMPT=' '):

    if AGENT_PROMPT.strip() == "":  
        AGENT_PROMPT = "."  
    messages.append({"role": "user", "content": AGENT_PROMPT})  

    agent_plan = llm_zhipu(messages)
    messages.append({"role": "assistant", "content": agent_plan})

    return agent_plan

