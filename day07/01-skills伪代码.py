# os：操作系统接口，用于读取环境变量
import os
# re：正则表达式模块，用于解析YAML frontmatter
import re
# json：JSON处理模块，用于解析工具调用参数
import json
# pathlib：面向对象的文件系统路径库
from pathlib import Path
# OpenAI：阿里云DashScope兼容的OpenAI客户端
from openai import OpenAI
# dotenv：从.env文件加载环境变量
from dotenv import load_dotenv

# ============================================================
# 配置部分    请先执行：uv add pytest
# ============================================================
# 从 .env 文件加载环境变量
load_dotenv()

# API配置
# DASHSCOPE_API_KEY：阿里云DashScope API密钥，用于身份验证
API_KEY = os.getenv("DASHSCOPE_API_KEY")
# DASHSCOPE_BASE_URL：API基础URL，如果使用自定义端点需要设置
BASE_URL = os.getenv("DASHSCOPE_BASE_URL")

# 技能目录路径
# 所有技能都存储在 /skills 目录下，每个技能一个子目录
SKILLS_DIR = Path("./skills")

# 使用的模型名称
# 这里使用阿里云DashScope的qwen3.5-plus模型
MODEL = "qwen3.5-plus"

# ============================================================
# 1. Skill 数据结构   name , description , content
# ============================================================
class Skill:
    """
    技能单元类，表示一个可调用的专业技能。

    每个技能对应一个 SKILL.md 文件，包含：
    - 技能名称（从目录名获取）
    - 技能描述（从 YAML frontmatter 解析）
    - 技能内容（SKILL.md 文件的完整内容）

    属性：
        name (str): 技能目录名称，例如 "frontend-design"
        description (str): 技能描述，用于工具描述，帮助模型理解何时使用该技能
        content (str): SKILL.md 文件的完整内容，作为工具调用的结果返回给模型
    """

    def __init__(self, name: str, description: str, content: str):
        """
        初始化技能单元。

        参数：
            name (str): 技能目录名称，例如 "frontend-design"
            description (str): 技能描述，从 YAML frontmatter 解析
            content (str): SKILL.md 文件的完整内容
        """
        self.name = name  # 技能目录名，如 "frontend-design"
        self.description = description  # 从 frontmatter 解析的描述，用于 Tool description
        self.content = content  # SKILL.md 完整内容，作为 tool_result 注入给模型

    # tostring
    def __repr__(self):
        """
        返回技能的字符串表示形式。

        返回：
            str: 技能的可读表示，例如 "Skill(name='frontend-design')"
        """
        return f"Skill(name={self.name!r})"


# ============================================================
# 2. Skill 加载器
# ============================================================

class SkillLoader:
    """
    技能加载器类，负责从文件系统加载所有技能。

    该类会扫描指定目录下的所有子目录，每个包含 SKILL.md 文件的子目录
    被视为一个独立的技能。加载器会解析每个技能的元数据（描述）和内容，
    并返回一个以工具名称为键的字典，便于快速查找。

    属性：
        skills_dir (Path): 技能目录的路径
    """

    def __init__(self, skills_dir: Path):
        """
        初始化技能加载器。

        参数：
            skills_dir (Path): 技能目录的路径，例如 Path("./skills")
        """
        self.skills_dir = skills_dir

    def load_all(self) -> dict[str, Skill]:
        """
        加载所有技能并返回技能字典。

        扫描 skills_dir 下的所有子目录，每个含有 SKILL.md 的子目录
        被视为一个 Skill。返回以 tool_name 为键的字典，便于 tool_use 时快速查找。

        返回：
            dict[str, Skill]: 技能字典，键为工具名称，值为 Skill 对象

        示例：
            ['skill_docx', 'skill_frontend_design', 'skill_xlsx']
        """
        skills: dict[str, Skill] = {}   # {"技能名称", 技能类}

        # 检查技能目录是否存在
        if not self.skills_dir.exists():
            print(f"技能目录不存在: {self.skills_dir}")
            return skills

        # 遍历技能目录下的所有子目录
        for skill_dir in sorted(self.skills_dir.iterdir()):
            # 跳过非目录文件
            if not skill_dir.is_dir():
                continue

            # 检查 SKILL.md 文件是否存在
            skill_md = skill_dir / "SKILL.md"
            if not skill_md.exists():
                continue

            # 读取 SKILL.md 文件内容
            content = skill_md.read_text(encoding="utf-8")

            # 从 YAML frontmatter 中解析技能描述
            description = self._parse_description(content)

            # 创建 Skill 对象
            skill = Skill(name=skill_dir.name, description=description, content=content)

            # 以 skill name 为键存储，便于 load_skill 时快速查找
            skills[skill.name] = skill
            print(f"加载: [{skill.name}]  →  description={skill.description[:30]}...")

        print(f"\n共加载 {len(skills)} 个 Skill\n")
        return skills

    @staticmethod
    def _parse_description(content: str) -> str:
        """
        从 SKILL.md 的 YAML frontmatter 中提取 description 字段。

        YAML frontmatter 是文件开头用 --- 包围的 YAML 块，
        包含技能的元数据，如名称、描述、版本等。

        参数：
            content (str): SKILL.md 文件的完整内容

        返回：
            str: 技能描述，如果解析失败则返回 "通用技能"

        示例：

            '创建具有高设计质量的独特的生产级前端界面。'
        """
        # 使用正则表达式匹配 YAML frontmatter 块
        # ^---\s*\n 匹配开头的 ---
        # (.*?)\n--- 匹配中间的内容直到下一个 ---
        # re.DOTALL 使 . 也匹配换行符
        m = re.match(r'^---\s*\n(.*?)\n---', content, re.DOTALL)
        if not m:
            return "通用技能"

        # 提取 YAML frontmatter 内容
        fm = m.group(1)

        # 从 YAML 中提取 description 字段
        # description:\s*["\']?(.+?)["\']?\s*$ 匹配 description: "value" 或 description: value
        dm = re.search(r'description:\s*["\']?(.+?)["\']?\s*$', fm, re.MULTILINE)

        # 返回描述，去除引号和首尾空格
        return dm.group(1).strip().strip('"\'') if dm else "通用技能"


# ============================================================
# 3. Agent 核心
# ============================================================

class LocalAgent:
    """
    基于 Tool Use 的智能代理类（opencode 风格）。

    这是整个系统的核心类，实现了基于工具调用的智能对话代理。
    它能够根据用户输入自动选择合适的技能（Skill），
    并结合技能指南生成专业的回答。

    属性：
        client (OpenAI): OpenAI 客户端实例
        skills (dict[str, Skill]): 技能字典，键为技能名称
        tool_definitions (list[dict]): 工具定义列表，用于 API 调用
        history (list[dict]): 对话历史记录

    工作机制（模拟 opencode 的 skills 机制）：
      1. 在系统提示词中列出所有可用的 skills 及其描述。
      2. 模型根据用户请求的语义判断需要哪个 skill。
      3. 模型调用 load_skill 工具，指定 skill 名称。
      4. Agent 收到工具调用后，读取对应 SKILL.md 内容返回给模型。
      5. 模型结合技能指南，生成最终专业回答。
    """

    # 系统提示词模板，{skills_list} 会被替换为实际的技能列表
    SYSTEM_PROMPT_TEMPLATE = """你是一个智能助手，拥有一组专业技能工具。

## 可用技能列表

{skills_list}

## 使用规则

- 当用户的请求涉及某个专业领域时，必须先调用 load_skill 工具加载对应的技能指南，然后再作答。
- 调用 load_skill 时，name 参数必须是上面列出的技能名称之一（如 "docx"、"frontend-design"、"xlsx"）。
- 获得技能指南后，严格遵循指南中的规范、最佳实践和代码风格。
- 如果任务不需要特定专业技能（如闲聊、一般知识问答），直接回答即可，无需调用工具。
- 一次请求可以调用多次 load_skill（如同时需要前端 + 文档技能）。
"""

    def __init__(self, api_key: str, base_url: str, skills_dir: Path):
        """
        初始化智能代理。

        参数：
            api_key (str): API 密钥，用于身份验证
            base_url (str): API 基础 URL，如果使用自定义端点需要设置
            skills_dir (Path): 技能目录的路径

        初始化过程：
            1. 创建 OpenAI 客户端实例
            2. 加载所有技能
            3. 构建技能描述列表（用于系统提示词）
            4. 构建工具定义列表（只有一个 load_skill 工具）
            5. 初始化对话历史
        """
        # 创建 OpenAI 客户端，用于与 API 通信
        self.client = OpenAI(api_key=api_key, base_url=base_url)

        # 加载所有 Skills，key = skill_name（如 "docx"、"frontend-design"）
        loader = SkillLoader(skills_dir)
        self.skills: dict[str, Skill] = loader.load_all()

        # 构建技能描述列表，用于系统提示词   获取对应的name和description
        skills_list = self._build_skills_list()

        # 动态生成系统提示词
        self.system_prompt = self.SYSTEM_PROMPT_TEMPLATE.format(
            skills_list=skills_list
        )

        # 构建 Tool Definitions 列表（只有一个 load_skill 工具）
        self.tool_definitions: list[dict] = [self._create_load_skill_tool()]

        # 多轮对话历史，存储用户和助手的交互记录
        self.history: list[dict] = []

    # ----------------------------------------------------------
    # 公开接口
    # ----------------------------------------------------------

    def chat(self, user_input: str) -> str:
        """
        处理一轮用户输入，返回最终回答。

        这是代理的核心方法，实现了完整的对话流程：
        1. 将用户输入添加到对话历史
        2. 调用 API 获取模型响应
        3. 如果模型选择调用工具，处理工具调用并获取结果
        4. 将工具结果返回给模型，继续对话
        5. 重复步骤 2-4，直到模型生成最终回答

        参数：
            user_input (str): 用户输入的文本

        返回：
            str: 代理的最终回答

        示例：

            '好的，我将使用前端设计技能帮你创建一个React组件...'
        """
        # 将用户消息加入历史
        self.history.append({"role": "user", "content": user_input})

        # Agentic Loop：持续处理，直到模型返回 end_turn（无更多工具调用）
        while True:
            # 调用 API 获取模型响应
            response = self._call_api()
            message = response.choices[0].message
            stop_reason = response.choices[0].finish_reason

            print(f"  [Agent] stop_reason={stop_reason!r}")

            # 根据停止原因处理响应    function call
            if stop_reason == "tool_calls":
                # 模型选择调用工具
                # 将本轮 assistant 消息加入历史
                self.history.append({
                    "role": "assistant",
                    "content": message.content or "",  # 可能为 None
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments  # JSON 字符串
                            }
                        }
                        for tc in (message.tool_calls or [])
                    ]
                })

                # 处理所有 tool_call，构建 tool 结果列表    # 代表skill的完整内容已经存在历史上下文里面
                tool_results = self._handle_tool_calls(message.tool_calls)

                # 每个 tool result 单独作为一条 tool 消息追加
                for result in tool_results:
                    self.history.append({
                        "role": "tool",
                        "tool_call_id": result["tool_call_id"],  # 必须与请求中的 id 对应
                        "content": result["content"]  # 字符串
                    })
                # 继续循环，让模型处理工具结果

            elif stop_reason == "stop":
                # 模型完成回答，没有更多工具调用
                final_text = message.content or ""

                # 将最终回答加入历史
                self.history.append({
                    "role": "assistant",
                    "content": final_text
                })
                return final_text

            else:
                # length 等其他停止原因
                return message.content or f"[停止原因: {stop_reason}]"

    def reset(self):
        """
        清空对话历史，重新开始对话。

        调用此方法会清除所有之前的对话记录，
        代理将从全新的状态开始。
        """
        self.history = []
        print("对话历史已清空\n")

    def show_tools(self):
        """
        显示所有已注册的技能和工具。

        打印所有可用的技能列表和工具定义，
        帮助用户了解代理具备哪些能力。
        """
        print("可用技能列表：")
        for skill in self.skills.values():
            print(f"  - {skill.name}: {skill.description}")
        print()

        print("已注册的工具：")
        for td in self.tool_definitions:
            print(f"  - {td['function']['name']}: {td['function']['description'][:50]}...")
        print()

    # ----------------------------------------------------------
    # 内部方法
    # ----------------------------------------------------------

    def _build_skills_list(self) -> str:
        """
        构建技能描述列表，用于系统提示词。

        将所有可用的技能以 Markdown 列表格式展示，
        让模型知道有哪些技能可用以及每个技能的用途。

        返回：
            str: 技能描述列表的 Markdown 文本

        示例返回值：
            - **docx**: 使用专业格式创建和操作Word文档（.docx文件）
            - **frontend-design**: 创建具有高设计质量的独特的生产级前端界面
            - **xlsx**: 使用openpyxl或pandas创建和操作Excel电子表格
        """
        lines = []
        for skill in self.skills.values():  # Skills对象
            lines.append(f"- **{skill.name}**: {skill.description}")
        return "\n".join(lines)

    def _create_load_skill_tool(self) -> dict:
        """
        创建 load_skill 工具定义。

        这是唯一的工具，用于按需加载指定 skill 的完整内容。
        模型通过调用此工具来获取技能指南。

        返回：
            dict: OpenAI 兼容的工具定义字典
        """
        return {
            "type": "function",
            "function": {
                "name": "load_skill",
                "description": (
                    "加载指定技能的完整指南。当用户的任务涉及某个专业领域时，"
                    "调用此工具加载对应的技能指南，然后严格按照指南中的规范完成任务。"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": (
                                    "要加载的技能名称，必须是以下之一："
                                    + "、".join(self.skills.keys())
                            )
                        }
                    },
                    "required": ["name"]
                }
            }
        }

    def _call_api(self):
        """
        调用模型 API，发送对话历史和工具定义。

        构建完整的请求消息，包括：
        1. 系统提示词（定义代理行为）
        2. 对话历史（用户和助手的交互）
        3. 工具定义（所有可用的技能工具）

        返回：
            API 响应对象，包含模型的回复和可能的工具调用
        """
        # 构建完整的消息列表
        messages = [
            {
                "role": "system",
                "content": self.system_prompt  # 使用动态生成的系统提示词
            },
            *self.history
        ]

        # 调用 API
        return self.client.chat.completions.create(
            model=MODEL,  # 使用的模型
            max_tokens=4096,  # 最大生成 token 数
            tools=self.tool_definitions,  # ← 所有 Skill 作为 Tools bind 到模型
            messages=messages,  # 对话历史
        )

    def _handle_tool_calls(self, tool_calls) -> list[dict]:
        """
        处理模型返回的工具调用请求。

        遍历所有工具调用请求，处理 load_skill 工具调用，
        返回对应的 SKILL.md 内容作为 tool message。

        参数：
            tool_calls: 模型返回的工具调用列表

        返回：
            list[dict]: 符合 OpenAI API 规范的工具结果列表

        返回格式：
            [
                {
                    "role": "tool",
                    "tool_call_id": "<对应 tool_call 的 id>",
                    "content": "<SKILL.md 内容>"
                },
                ...
            ]
        """
        import json
        tool_results = []

        # 遍历所有工具调用请求
        for tc in (tool_calls or []):
            tool_name = tc.function.name
            # 解析工具参数（OpenAI 格式是 JSON 字符串）
            tool_input = json.loads(tc.function.arguments)

            # 处理 load_skill 工具调用
            if tool_name == "load_skill":
                skill_name = tool_input.get("name", "")
                print(f"  [Tool Call] 加载技能: {skill_name!r}")

                # 查找对应的技能
                skill = self.skills.get(skill_name)

                if skill:
                    # 构建技能指南内容
                    result_content = (
                        f"# 技能指南已加载：{skill.name}\n\n"
                        f"{skill.content}\n\n"
                        "---\n请严格按照以上指南完成任务。"
                    )
                    print(f"→ 注入 [{skill.name}]，{len(skill.content)} 字符")
                else:
                    # 技能未找到，列出可用的技能
                    available = ", ".join(self.skills.keys())
                    result_content = (
                        f"错误：未找到名为 {skill_name!r} 的技能。\n\n"
                        f"可用的技能有：{available}"
                    )
                    print(f" → 未找到技能: {skill_name!r}")
            else:
                # 未知的工具调用
                result_content = f"错误：未知的工具 {tool_name!r}。请使用 load_skill 工具加载技能。"
                print(f"  [Tool Call] 未知工具: {tool_name!r}")

            # 构建工具结果
            tool_results.append({
                "role": "tool",  # ← Anthropic 是 "type": "tool_result"
                "tool_call_id": tc.id,  # ← Anthropic 是 "tool_use_id": block.id
                "content": result_content,
            })

        return tool_results

    @staticmethod
    def _extract_text(content_blocks: list) -> str:
        """
        从 content block 列表中提取所有文本内容。

        这是一个辅助方法，用于从复杂的响应结构中提取纯文本。
        支持两种格式：
        1. 具有 text 属性的对象
        2. 包含 type="text" 的字典

        参数：
            content_blocks (list): 内容块列表

        返回：
            str: 提取的文本内容，用换行符连接
        """
        parts = []
        for block in content_blocks:
            # 处理对象格式
            if hasattr(block, "text"):
                parts.append(block.text)
            # 处理字典格式
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block["content"])
        return "\n".join(parts).strip()


# ============================================================
# 4. 主程序
# ============================================================

def main():
    """
    主程序入口函数。

    这个函数实现了交互式命令行界面，允许用户与智能代理进行对话。
    程序流程：
    1. 显示欢迎信息
    2. 初始化智能代理
    3. 显示可用的技能工具
    4. 检查 API 密钥配置
    5. 进入交互式对话循环

    支持的命令：
    - 'reset'：清空对话历史，重新开始
    - 'quit'：退出程序
    - Ctrl+C：中断程序

    使用方法：
        直接运行 python skills_agent.py 即可启动交互式界面
    """
    # 显示欢迎信息
    print("=" * 60)
    print("   Skills Agent  (Tool Use 版)")
    print("=" * 60)
    print()

    # 初始化智能代理
    # 使用配置文件中的 API 密钥、基础 URL 和技能目录
    agent = LocalAgent(api_key=API_KEY, base_url=BASE_URL, skills_dir=SKILLS_DIR)

    # 显示所有可用的技能工具
    agent.show_tools()

    # 检查 API 密钥是否已配置
    if API_KEY == "your-api-key-here":
        print("未设置 ANTHROPIC_API_KEY，仅展示 Tool Definition 结构\n")
        # 显示工具定义的 JSON 结构，便于调试
        print(json.dumps(agent.tool_definitions, ensure_ascii=False, indent=2))
        print("\n请设置后重新运行：  export ANTHROPIC_API_KEY=sk-ant-...")
        return

    # 显示使用说明
    print("输入 'reset' 清空历史，'quit' 退出\n")

    # 交互式对话循环
    while True:
        try:
            # 获取用户输入
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            # 处理 Ctrl+C 或 EOF 信号
            print("\nGoodbye!")
            break

        # 跳过空输入
        if not user_input:
            continue

        # 处理退出命令
        if user_input.lower() == "quit":
            print("Goodbye!")
            break

        # 处理重置命令
        if user_input.lower() == "reset":
            agent.reset()
            continue

        # 处理用户输入，获取代理回答
        print()
        answer = agent.chat(user_input)
        print(f"\nAssistant:\n{answer}\n")


# ============================================================
# 程序入口点
# ============================================================
# 当直接运行此脚本时，调用 main() 函数启动交互式界面
# 如果作为模块导入，则不会自动执行
if __name__ == "__main__":
    main()