
from zhipuai import ZhipuAI
def llm_zhipu(PROMPT=' '):

    API_KEY = " "  # 请填写您自己的APIKey
    MODEL = "glm-4-flash"
    client = ZhipuAI(api_key=API_KEY)  
    response = client.chat.completions.create(
        model=MODEL,
        messages=PROMPT,
        max_tokens=1024,
    )
 
    result = response.choices[0].message.content.strip()
    return result
    
