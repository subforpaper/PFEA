
import base64
from zhipuai import ZhipuAI


PROMPT_Stage1_Attribute = '''
你是Chain-of-Objects Reasoning(CoOR)的第一阶段：属性提取与物体定位助手。
任务：观察图片，识别所有任务相关物体，提取每个物体的属性向量φ(o_i)=[color_i, shape_i, size_i, co_i(几何属性如角数), r_i(空间关系)]

【输出json格式】
你直接输出json即可，从{开始，输出中不要包含json。
输出格式：
{
'stage': 1,
'stage_name': 'Attribute Extraction and Object Grounding',
'objects': {
'物体描述1': {'color': '颜色', 'shape': '形状', 'corners': 角数, 'size': '尺寸描述', 'position': '位置描述'},
'物体描述2': {'color': '颜色', 'shape': '形状', 'corners': 角数, 'size': '尺寸描述', 'position': '位置描述'}
},
'task_instruction': '复述任务指令'
}

【示例】
指令：按照物体角的多少从多到少依次堆叠。
你输出：
{
'stage': 1,
'stage_name': 'Attribute Extraction and Object Grounding',
'objects': {
'黄色六边形': {'color': '黄色', 'shape': '六边形', 'corners': 6, 'size': '中等', 'position': '桌面中央'},
'绿色五角星': {'color': '绿色', 'shape': '五角星', 'corners': 5, 'size': '中等', 'position': '六边形左侧'},
'蓝色四边形': {'color': '蓝色', 'shape': '四边形', 'corners': 4, 'size': '中等', 'position': '桌面右侧'},
'紫色三角形': {'color': '紫色', 'shape': '三角形', 'corners': 3, 'size': '中等', 'position': '桌面边缘'}
},
'task_instruction': '按照物体角的多少从多到少依次堆叠'
}

【我现在的指令是】
'''

PROMPT_Stage2_Ordering = '''
你是Chain-of-Objects Reasoning(CoOR)的第二阶段：链式推理排序助手。
任务：根据第一阶段的物体属性提取结果，结合任务指令，建立执行顺序σ，定义排序函数σ(o_i)=rank(co_i, C)

【重要规则：堆叠任务的通用逻辑】
对于任何堆叠任务（Stack objects），你需要理解以下规则：
- 任务指令会明确说明哪个物体在最下面（底座）
- 底座物体不需要被移动，放在桌面上不动
- 执行顺序：从底座上面的第一个物体开始，依次放到下面的物体上
- 第1个动作：将第2层物体 放到 底座物体 上
- 第2个动作：将第3层物体 放到 第2层物体 上
- 第3个动作：将第4层物体 放到 第3层物体 上
- 简单记：下面的物体是"地基"，上面的物体往下面放

【关键理解示例】
示例1：指令"按照物体角的多少从多到少依次堆叠"
- 堆叠顺序（从下到上）：六边形(6角) → 五角星(5角) → 四边形(4角) → 三角形(3角)
- 底座：六边形(不动)
- 执行：将五角星放到六边形上 → 将四边形放到五角星上 → 将三角形放到四边形上

示例2：指令"按字母A到Z的顺序，A在最下面，将字母堆叠起来"
- 堆叠顺序（从下到上）：A → G → R → V
- 底座：A(不动)
- 执行：将G放到A上 → 将R放到G上 → 将V放到R上

【输入】
第一阶段输出：
{STAGE1_OUTPUT}

【输出json格式】
你直接输出json即可，从{开始，输出中不要包含json。
输出格式：
{
'stage': 2,
'stage_name': 'Sequential Ordering via Chain Reasoning',
'ordering_criteria': '排序依据（如角数降序、字母顺序等）',
'stacking_order_bottom_to_top': '底座物体→第二层物体→第三层物体→...→最顶层物体',
'base_object': '底座物体名称（在最下面，不需要被移动）',
'execution_sequence': ['第二层物体', '第三层物体', '第四层物体'],
'reasoning': '简述排序推理过程，明确说明哪个是底座，然后依次将哪些物体放到上面'
}

【示例】
第一阶段输出：
{
'stage': 1,
'stage_name': 'Attribute Extraction and Object Grounding',
'objects': {
'黄色六边形': {'color': '黄色', 'shape': '六边形', 'corners': 6, 'size': '中等', 'position': '桌面中央'},
'绿色五角星': {'color': '绿色', 'shape': '五角星', 'corners': 5, 'size': '中等', 'position': '六边形左侧'},
'蓝色四边形': {'color': '蓝色', 'shape': '四边形', 'corners': 4, 'size': '中等', 'position': '桌面右侧'},
'紫色三角形': {'color': '紫色', 'shape': '三角形', 'corners': 3, 'size': '中等', 'position': '桌面边缘'}
},
'task_instruction': '按照物体角的多少从多到少依次堆叠'
}
你输出：
{
'stage': 2,
'stage_name': 'Sequential Ordering via Chain Reasoning',
'ordering_criteria': '按角数降序排列（corners: most to fewest）',
'stacking_order_bottom_to_top': '六边形(6)→五角星(5)→四边形(4)→三角形(3)',
'base_object': '黄色六边形',
'execution_sequence': ['绿色五角星', '蓝色四边形', '紫色三角形'],
'reasoning': '指令"从多到少堆叠"意味着：角数最多的六边形(6)作为底座放在最下面，然后依次往上堆叠。具体步骤：(1)将绿色五角星(5)放到黄色六边形上，(2)将蓝色四边形(4)放到绿色五角星上，(3)将紫色三角形(3)放到蓝色四边形上。'
}

【请根据第一阶段输出和任务指令进行推理】
'''

PROMPT_Stage3_Grounding = '''
你是Chain-of-Objects Reasoning(CoOR)的第三阶段：空间关系定位助手。
任务：根据第两阶段的推理结果，确定每个步骤的源物体、目标物体和空间关系R(s_j)=(o_source, o_target, r)，其中r∈{on, into, beside}，最终生成可执行的机器人操作指令。

【重要规则：堆叠任务的通用逻辑】
对于任何堆叠任务（Stack objects）：
- source是要被抓取的物体（需要被移动的）
- target是目标位置的物体（作为放置目标的）
- 堆叠顺序（从下到上）：A→B→C→D，其中A是底座不动
- 第1步：将B放到A上 → source='B', target='A'
- 第2步：将C放到B上 → source='C', target='B'（注意target是刚才放上去的B）
- 第3步：将D放到C上 → source='D', target='C'
- 核心原则：每次放的target是下面那层的物体（即execution_sequence中前一个物体，或base_object）

【关键理解】
如果stacking_order_bottom_to_top是：三角形(3)→正方形(4)→五角星(5)→六边形(6)
- 底座（不动）：三角形
- 执行顺序：正方形→五角星→六边形
- 正确操作：
  - 第1步：将正方形放到三角形上 → source='正方形', target='三角形'
  - 第2步：将五角星放到正方形上 → source='五角星', target='正方形'
  - 第3步：将六边形放到五角星上 → source='六边形', target='五角星'

【输入】
第二阶段排序推理：
{STAGE2_OUTPUT}

【输出json格式】
你直接输出json即可，从{开始，输出中不要包含json。
输出格式：
{
'stage': 3,
'stage_name': 'Spatial Relationship Grounding',
'spatial_relationships': [
{'step': 1, 'source': '源物体', 'target': '目标物体', 'relation': 'on/into/beside', 'action_description': '动作描述'},
{'step': 2, 'source': '源物体', 'target': '目标物体', 'relation': 'on/into/beside', 'action_description': '动作描述'}
],
'function': ['将物体A放到物体B上/里', '将物体C放到物体D上/里']
}

【示例：按照物体角的数量从多到少依次堆叠物体】
第二阶段输出：
{
'stage': 2,
'stage_name': 'Sequential Ordering via Chain Reasoning',
'ordering_criteria': '按角数降序排列（corners: most to fewest）',
'stacking_order_bottom_to_top': '六边形(6)→五角星(5)→四边形(4)→三角形(3)',
'base_object': '黄色六边形',
'execution_sequence': ['绿色五角星', '蓝色四边形', '紫色三角形'],
'reasoning': '指令"从多到少堆叠"意味着：角数最多的六边形(6)作为底座放在最下面，然后依次往上堆叠。具体步骤：(1)将绿色五角星(5)放到黄色六边形上，(2)将蓝色四边形(4)放到绿色五角星上，(3)将紫色三角形(3)放到蓝色四边形上。'
}

你输出：
{
'stage': 3,
'stage_name': 'Spatial Relationship Grounding',
'spatial_relationships': [
{'step': 1, 'source': '绿色五角星', 'target': '黄色六边形', 'relation': 'on', 'action_description': '将五角星堆叠到六边形上'},
{'step': 2, 'source': '蓝色四边形', 'target': '绿色五角星', 'relation': 'on', 'action_description': '将四边形堆叠到五角星上'},
{'step': 3, 'source': '紫色三角形', 'target': '蓝色四边形', 'relation': 'on', 'action_description': '将三角形堆叠到四边形上'}
],
'function': ['将绿色的五角星放到黄色的六边形上', '将蓝色四边形放到绿色五角星上', '将紫色的三角形放到蓝色四边形上']
}

【请根据前两阶段输出进行推理】
'''

PROMPT_back01 = '''
【反馈控制】
你是一个反馈控制助手。你需要根据图片信息，判断任务是否完成。
观察图片，物体是否集中到一起，若集中到一起请输出：{'function':['任务完成']}，若有零散的物体请输出：{'function':['任务未完成']}。
观察图片，物体是否都放到一个盒子里，若放到一个盒子里请输出：{'function':['任务完成']}，否则请输出：{'function':['任务未完成']}。
【输出json格式】 
你直接输出json即可，从{开始，输出中不要包含json，在'function'键中，输出函数名列表，列表中每个元素都是字符串，代表要运行的具体语句。
{TEXT}：
'''


def vlm_coOR(TEXT, img_path):

    API_KEY = " "  # 请填写您自己的APIKey
    MODEL = "glm-4v-flash"
    client = ZhipuAI(api_key=API_KEY)
    
    with open(img_path, 'rb') as img_file:
        img_base = base64.b64encode(img_file.read()).decode('utf-8')
    
    def call_vlm(prompt):
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": img_base
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
        )
        return response.choices[0].message.content.strip()
    
    stage1_prompt = PROMPT_Stage1_Attribute + f"\n指令：{TEXT}"
    stage1_result = call_vlm(stage1_prompt)
    print(f"Stage 1:\n{stage1_result}")

    stage2_prompt = PROMPT_Stage2_Ordering.replace('{STAGE1_OUTPUT}', stage1_result)
    stage2_prompt += f"\n指令：{TEXT}"
    stage2_result = call_vlm(stage2_prompt)
    print(f"Stage 2:\n{stage2_result}")

    stage3_prompt = PROMPT_Stage3_Grounding.replace('{STAGE2_OUTPUT}', stage2_result)
    stage3_prompt += f"\n指令：{TEXT}"
    stage3_result = call_vlm(stage3_prompt)
    print(f"Stage 3:\n{stage3_result}")
    
    
    return stage3_result


def vlm_ROF(PROMPT, TEXT, img_path):
    
    API_KEY = " "  # 请填写您自己的APIKey
    MODEL = "glm-4v-flash"
    client = ZhipuAI(api_key=API_KEY)  
    with open(img_path, 'rb') as img_file:
        img_base = base64.b64encode(img_file.read()).decode('utf-8')
    response = client.chat.completions.create(
        model=MODEL,

        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": img_base
                        }
                    },
                    {
                        "type": "text",
                        "text": PROMPT + TEXT
                    }
                ]
            }
        ]

    )
 
    result = response.choices[0].message.content.strip()
    return result
