# 智能体经典范式构建

为了解决智能体面临的来自大模型本身的“幻觉”问题、在复杂任务中可能陷入推理循环、以及对工具的错误使用等挑战，出现了几种范式：

- **ReAct (Reasoning and Acting)**： 一种将“思考”和“行动”紧密结合的范式，让智能体边想边做，动态调整。
- **Plan-and-Solve**： 一种“三思而后行”的范式，智能体首先生成一个完整的行动计划，然后严格执行。
- **Reflection**： 一种赋予智能体“反思”能力的范式，通过自我批判和修正来优化结果。

## ReAct范式

之前两种大模型工作模式：

1. **思维链型**：引导模型进行复杂的逻辑推理，但无法与外部世界进行交互，容易出现幻觉；
2. **行动型**：直接输出模型要执行的结果，但是缺乏规划和纠错能力；

ReAct工作范式是思考+行动相辅相成，即：

- **thought**：分析情况、分解任务、指定下一步计划，或者反思上一步结果；
- **action**：调用工具，实现一个具体操作；
- **observation**：工具的返回结果；

智能体不断重复thought->action->observation过程，并不断把之前的结果加入到上下文中，指导下一次action，直到它在thought中认为已经找到了答案，然后输出最终结果。形式化的表达，则是：
当前时间 $$t$$ 下的思考 $$th_{t}$$ 和行动 $$a_{t}$$ 为

$$
\left( th_t , a_t \right) = \pi (q, (a_1,o_1), (a_2,o_2), ..., (a_{t-1}，o_{t-1}))
$$

### ReAct智能体实现

#### 系统提示词设计

```
# ReAct 提示词模板
REACT_PROMPT_TEMPLATE = """
请注意，你是一个有能力调用外部工具的智能助手。

可用工具如下:
{tools}

请严格按照以下格式进行回应:

Thought: 你的思考过程，用于分析问题、拆解任务和规划下一步行动。
Action: 你决定采取的行动，必须是以下格式之一:
- `{{tool_name}}[{{tool_input}}]`:调用一个可用工具。
- `Finish[最终答案]`:当你认为已经获得最终答案时。
- 当你收集到足够的信息，能够回答用户的最终问题时，你必须在Action:字段后使用 Finish[最终答案] 来输出最终答案。

现在，请开始解决以下问题:
Question: {question}
History: {history}
"""
```

#### 核心循环设计

ReAct范式的智能体核心是一个循环：格式化提示词->调用LLM->执行动作->整合结果，直到达到最大步数。

```python
class ReActAgent:
    def __init__(self, llm_client: HelloAgentsLLM, tool_executor: ToolExecutor, max_steps: int = 5):
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.max_steps = max_steps
        self.history = []

    def run(self, question: str):
        """
        运行ReAct智能体来回答一个问题。
        """
        self.history = [] # 每次运行时重置历史记录
        current_step = 0

        while current_step < self.max_steps:
            current_step += 1
            print(f"--- 第 {current_step} 步 ---")

            # 1. 格式化提示词
            tools_desc = self.tool_executor.getAvailableTools()
            history_str = "\n".join(self.history)
            prompt = REACT_PROMPT_TEMPLATE.format(
                tools=tools_desc,
                question=question,
                history=history_str
            )

            # 2. 调用LLM进行思考
            messages = [{"role": "user", "content": prompt}]
            response_text = self.llm_client.think(messages=messages)
            
            if not response_text:
                print("错误:LLM未能返回有效响应。")
                break

            # ... (后续的解析、执行、整合步骤)
```

`run`是整个智能体的入口，`while`构成ReAct范式智能体的主体，`max_steps`是为了防止智能体出现无限循环。

#### 输出解析器的实现

从`output`中解析出 `thought` 和 `action`，以及从`action`中解析出工具和工具入参

```python
# (这些方法是 ReActAgent 类的一部分)
    def _parse_output(self, text: str):
        """解析LLM的输出，提取Thought和Action。
        """
        # Thought: 匹配到 Action: 或文本末尾
        thought_match = re.search(r"Thought:\s*(.*?)(?=\nAction:|$)", text, re.DOTALL)
        # Action: 匹配到文本末尾
        action_match = re.search(r"Action:\s*(.*?)$", text, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else None
        action = action_match.group(1).strip() if action_match else None
        return thought, action

    def _parse_action(self, action_text: str):
        """解析Action字符串，提取工具名称和输入。
        """
        match = re.match(r"(\w+)\[(.*)\]", action_text, re.DOTALL)
        if match:
            return match.group(1), match.group(2)
        return None, None
```

#### 工具调用与执行

首先判断是否找到最终答案，否则执行工具。

```python
# (这段逻辑在 run 方法的 while 循环内)
            # 3. 解析LLM的输出
            thought, action = self._parse_output(response_text)
            
            if thought:
                print(f"思考: {thought}")

            if not action:
                print("警告:未能解析出有效的Action，流程终止。")
                break

            # 4. 执行Action
            if action.startswith("Finish"):
                # 如果是Finish指令，提取最终答案并结束
                final_answer = re.match(r"Finish\[(.*)\]", action).group(1)
                print(f"🎉 最终答案: {final_answer}")
                return final_answer
            
            tool_name, tool_input = self._parse_action(action)
            if not tool_name or not tool_input:
                # ... 处理无效Action格式 ...
                continue

            print(f"🎬 行动: {tool_name}[{tool_input}]")
            
            tool_function = self.tool_executor.getTool(tool_name)
            if not tool_function:
                observation = f"错误:未找到名为 '{tool_name}' 的工具。"
            else:
                observation = tool_function(tool_input) # 调用真实工具
```

#### 观测结果的整合

将本轮的Action和Observation添加到历史记录中

```python
# (这段逻辑紧随工具调用之后，在 while 循环的末尾)
            print(f"👀 观察: {observation}")
            
            # 将本轮的Action和Observation添加到历史记录中
            self.history.append(f"Action: {action}")
            self.history.append(f"Observation: {observation}")

        # 循环结束
        print("已达到最大步数，流程终止。")
        return None
```

### ReAct范式的优缺点

优点：

- 高可解释性：ReAct范式智能体将大模型的思维过程展示出来，能够清晰的看到大模型这一步为什么这么做、下一步要做什么；
- 动态规划和纠错能力：“走一步，思考一步”的方式，让智能体能够根据当前的外部行动结果及时调整下一步的行动；
- 工具协同能力：将大模型的推理能力与外部工具调用能力天然结合在一起。

有限性：

- 提示词脆弱性：整个ReAct的执行机制建立在一个精心设计的提示词模板上，ReAct智能体执行结果的好坏，很大程度上依赖这个提示词模板；
- 局部最优：“步进式”的思考导致大模型缺少一个全局、长远的规划，使大模型陷入看似正确但长远来看并非最优的路径；

## Plan-and-Solve范式

相比于ReAct范式，Plan-and-Solve范式是先规划（plan），然后再执行（solve）：

- 规划（plan）：智能体接收到用户的问题，首先对问题进行分解，制定一个清晰的，分步执行的实施计划。这个计划本身就是大语言模型调用产物；
- 执行（solve）：智能体严格按照实施计划进行处理，直到最后一步，并输出结果；

用数学公式直观表示，规划模型 $$\pi_{plan}$$ 根据原始问题 $$q$$ ，生成一个包含 $$n$$ 步的计划 $$P=(p_{1},...,p_{n})$$ :

$$
P=\pi_{plan}(q)
$$

行动模型 $$\pi_{solve}$$，依据原始问题 $$q$$、完整计划 $$P$$，以及前 $$i-1$$步行动结果，得到第 $$i$$ 步的结果为：

$$
s_{i} = \pi_{solve}(q,P,(s_{1},...,s_{i-1}))
$$

最终答案，是最后一步的结果 $$s_{n}$$。

### 智能体实现

#### plan阶段

规划阶段是需要大模型根据原始问题，输出一份清晰，分步执行的实施方案。因此，设计的提示词需要明确地告诉模型它的角色和任务：

````python
PLANNER_PROMPT_TEMPLATE = """
你是一个顶级的AI规划专家。你的任务是将用户提出的复杂问题分解成一个由多个简单步骤组成的行动计划。
请确保计划中的每个步骤都是一个独立的、可执行的子任务，并且严格按照逻辑顺序排列。
你的输出必须是一个Python列表，其中每个元素都是一个描述子任务的字符串。

问题: {question}

请严格按照以下格式输出你的计划,```python与```作为前后缀是必要的:
\```python
["步骤1", "步骤2", "步骤3", ...]
```
"""
````

这个提示词通过以下几点确保了输出的质量和稳定性：

- 角色设定： “顶级的AI规划专家”，激发模型的专业能力。
- 任务描述： 清晰地定义了“分解问题”的目标。
- 格式约束： 强制要求输出为一个 Python 列表格式的字符串，这极大地简化了后续代码的解析工作，使其比解析自然语言更稳定、更可靠。

接下来将这个提示词封装成一个 `Planner`类：

```python
# 假定 llm_client.py 中的 HelloAgentsLLM 类已经定义好
# from llm_client import HelloAgentsLLM

class Planner:
    def __init__(self, llm_client):
        self.llm_client = llm_client

    def plan(self, question: str) -> list[str]:
        """
        根据用户问题生成一个行动计划。
        """
        prompt = PLANNER_PROMPT_TEMPLATE.format(question=question)
        
        # 为了生成计划，我们构建一个简单的消息列表
        messages = [{"role": "user", "content": prompt}]
        
        print("--- 正在生成计划 ---")
        # 使用流式输出来获取完整的计划
        response_text = self.llm_client.think(messages=messages) or ""
        
        print(f"✅ 计划已生成:\n{response_text}")
        
        # 解析LLM输出的列表字符串
        try:
            # 找到```python和```之间的内容
            plan_str = response_text.split("```python")[1].split("```")[0].strip()
            # 使用ast.literal_eval来安全地执行字符串，将其转换为Python列表
            plan = ast.literal_eval(plan_str)
            return plan if isinstance(plan, list) else []
        except (ValueError, SyntaxError, IndexError) as e:
            print(f"❌ 解析计划时出错: {e}")
            print(f"原始响应: {response_text}")
            return []
        except Exception as e:
            print(f"❌ 解析计划时发生未知错误: {e}")
            return []
```

#### solve阶段

执行器（Executor）调用大语言模型来解决每一个子任务，同时还要负责 **状态管理**，即负责记录每一步的结果，将其作为上下文提供给后续步骤。

##### 系统提示词设计

提示词需要包含以下关键信息：

- 原始问题： 确保模型始终了解最终目标。
- 完整计划： 让模型了解当前步骤在整个任务中的位置。
- 历史步骤与结果： 提供至今为止已经完成的工作，作为当前步骤的直接输入。
- 当前步骤： 明确指示模型现在需要解决哪一个具体任务。

```
EXECUTOR_PROMPT_TEMPLATE = """
你是一位顶级的AI执行专家。你的任务是严格按照给定的计划，一步步地解决问题。
你将收到原始问题、完整的计划、以及到目前为止已经完成的步骤和结果。
请你专注于解决“当前步骤”，并仅输出该步骤的最终答案，不要输出任何额外的解释或对话。

# 原始问题:
{question}

# 完整计划:
{plan}

# 历史步骤与结果:
{history}

# 当前步骤:
{current_step}

请仅输出针对“当前步骤”的回答:
"""
```

##### 执行器设计

```python
class Executor:
    def __init__(self, llm_client):
        self.llm_client = llm_client

    def execute(self, question: str, plan: list[str]) -> str:
        """
        根据计划，逐步执行并解决问题。
        """
        history = "" # 用于存储历史步骤和结果的字符串
        
        print("\n--- 正在执行计划 ---")
        
        for i, step in enumerate(plan):
            print(f"\n-> 正在执行步骤 {i+1}/{len(plan)}: {step}")
            
            prompt = EXECUTOR_PROMPT_TEMPLATE.format(
                question=question,
                plan=plan,
                history=history if history else "无", # 如果是第一步，则历史为空
                current_step=step
            )
            
            messages = [{"role": "user", "content": prompt}]
            
            response_text = self.llm_client.think(messages=messages) or ""
            
            # 更新历史记录，为下一步做准备
            history += f"步骤 {i+1}: {step}\n结果: {response_text}\n\n"
            
            print(f"✅ 步骤 {i+1} 已完成，结果: {response_text}")

        # 循环结束后，最后一步的响应就是最终答案
        final_answer = response_text
        return final_answer
```

##### 统一Planner和Executor

```python
class PlanAndSolveAgent:
    def __init__(self, llm_client):
        """
        初始化智能体，同时创建规划器和执行器实例。
        """
        self.llm_client = llm_client
        self.planner = Planner(self.llm_client)
        self.executor = Executor(self.llm_client)

    def run(self, question: str):
        """
        运行智能体的完整流程:先规划，后执行。
        """
        print(f"\n--- 开始处理问题 ---\n问题: {question}")
        
        # 1. 调用规划器生成计划
        plan = self.planner.plan(question)
        
        # 检查计划是否成功生成
        if not plan:
            print("\n--- 任务终止 --- \n无法生成有效的行动计划。")
            return

        # 2. 调用执行器执行计划
        final_answer = self.executor.execute(question, plan)
        
        print(f"\n--- 任务完成 ---\n最终答案: {final_answer}")
```

### Reflection范式

