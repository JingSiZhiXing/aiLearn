from langchain_core.utils.uuid import uuid7
from typing import TypedDict
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse, AgentMiddleware
from langchain.messages import SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from typing import Callable


# 定义技能结构
class Skill(TypedDict):
    """支持按需加载的技能定义"""
    name: str  # 技能名称
    description: str  # 技能简介
    content: str  # 技能详细内容


# 定义技能（Schema + 业务规则）
SKILLS: list[Skill] = [
    {
        "name": "sales_analytics",
        "description": "销售数据分析相关数据库结构与业务规则，包括客户、订单、收入统计。",
        "content": """
# 销售分析数据库说明

## 数据表

### customers（客户表）
- customer_id（主键）
- name（客户姓名）
- email（邮箱）
- signup_date（注册日期）
- status（状态：active/inactive）
- customer_tier（客户等级：bronze/silver/gold/platinum）

### orders（订单表）
- order_id（主键）
- customer_id（外键 -> customers）
- order_date（下单时间）
- status（状态：pending/completed/cancelled/refunded）
- total_amount（订单总金额）
- sales_region（销售区域：north/south/east/west）

### order_items（订单明细表）
- item_id（主键）
- order_id（外键 -> orders）
- product_id（商品ID）
- quantity（购买数量）
- unit_price（单价）
- discount_percent（折扣百分比）

---

## 业务规则

【活跃客户】
status='active'
且注册时间早于当前日期 90 天。

【收入统计】
仅统计 status='completed' 的订单，
收入直接使用 orders.total_amount。

【客户生命周期价值（CLV）】
统计客户所有已完成订单金额总和。

【高价值订单】
订单金额 total_amount > 1000。

---

## 示例 SQL

-- 查询最近一个季度收入最高的前10名客户
SELECT
    c.customer_id,
    c.name,
    c.customer_tier,
    SUM(o.total_amount) AS total_revenue
FROM customers c
JOIN orders o
ON c.customer_id = o.customer_id
WHERE o.status='completed'
AND o.order_date >= CURRENT_DATE - INTERVAL '3 months'
GROUP BY
    c.customer_id,
    c.name,
    c.customer_tier
ORDER BY total_revenue DESC
LIMIT 10;
""",
    },
    {
        "name": "inventory_management",
        "description": "库存管理数据库结构与业务规则，包括商品、仓库和库存分析。",
        "content": """
# 库存管理数据库说明

## 数据表

### products（商品表）
- product_id（主键）
- product_name（商品名称）
- sku（库存编码）
- category（分类）
- unit_cost（成本价）
- reorder_point（补货阈值）
- discontinued（是否停售）

### warehouses（仓库表）
- warehouse_id（主键）
- warehouse_name（仓库名称）
- location（位置）
- capacity（容量）

### inventory（库存表）
- inventory_id（主键）
- product_id（外键 -> products）
- warehouse_id（外键 -> warehouses）
- quantity_on_hand（现有库存）
- last_updated（更新时间）

### stock_movements（库存流水）
- movement_id（主键）
- product_id（外键）
- warehouse_id（外键）
- movement_type（类型：
  inbound 入库 /
  outbound 出库 /
  transfer 调拨 /
  adjustment 调整）
- quantity（数量）
- movement_date（发生时间）
- reference_number（单据号）

---

## 业务规则

【可用库存】
inventory.quantity_on_hand > 0。

【需要补货商品】
所有仓库库存总和
<= 产品 reorder_point。

【有效商品】
默认排除停售商品：
discontinued=false。

【库存估值】
quantity_on_hand × unit_cost。

---

## 示例 SQL

-- 查询库存低于补货线的商品
SELECT
    p.product_id,
    p.product_name,
    p.reorder_point,
    SUM(i.quantity_on_hand) AS total_stock,
    p.unit_cost,
    (
        p.reorder_point
        - SUM(i.quantity_on_hand)
    ) AS units_to_reorder
FROM products p
JOIN inventory i
ON p.product_id=i.product_id
WHERE p.discontinued=false
GROUP BY
    p.product_id,
    p.product_name,
    p.reorder_point,
    p.unit_cost
HAVING
    SUM(i.quantity_on_hand)
    <= p.reorder_point
ORDER BY units_to_reorder DESC;
""",
    },
]


# 创建技能加载工具
@tool
def load_skill(skill_name: str) -> str:
    """
    加载指定技能的完整内容到上下文。

    当需要详细规则、数据库结构、
    SQL 编写规范时调用。

    参数：
        skill_name：技能名称
    """

    for skill in SKILLS:
        if skill["name"] == skill_name:
            return (
                f"已加载技能：{skill_name}\n\n"
                f"{skill['content']}"
            )

    available = ", ".join(
        s["name"]
        for s in SKILLS
    )

    return (
        f"未找到技能：{skill_name}\n"
        f"可用技能：{available}"
    )


# 技能中间件     - 在调用模型之前讲系统中所有的skills的name和描述放到提示词中
class SkillMiddleware(AgentMiddleware):
    """向系统提示中注入技能目录"""

    tools = [load_skill]

    def __init__(self):
        skills_list = []

        for skill in SKILLS:
            skills_list.append(
                f"- {skill['name']}："
                f"{skill['description']}"
            )

        self.skills_prompt = "\n".join(
            skills_list
        )

    def wrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[
                [ModelRequest],
                ModelResponse
            ],
    ) -> ModelResponse:
        skills_addendum = f"""
        ## 可用技能
        {self.skills_prompt}
        如需完整规则，请调用：
        load_skill
        """
        new_content = (
                list(
                    request.system_message.content_blocks
                )
                + [
                    {
                        "type": "text",
                        "text": skills_addendum
                    }
                ]
        )

        new_system_message = SystemMessage(
            content=new_content
        )

        modified_request = (
            request.override(
                system_message=new_system_message
            )
        )

        return handler(
            modified_request
        )


# 初始化模型
import os
from dotenv import load_dotenv
from langchain_qwq import ChatQwen

load_dotenv()

model = ChatQwen(
    model="qwen3.7-max-2026-05-20",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
    enable_thinking=False,
)

# 创建 Agent
agent = create_agent(
    model=model,
    system_prompt="""
你是一名 SQL 查询助手。

你的职责：
1. 理解用户业务需求
2. 按需加载技能
3. 根据数据库结构生成 SQL
4. 遵循业务规则
""",
    middleware=[
        SkillMiddleware()
    ],
    checkpointer=InMemorySaver(),
)

# 测试运行
if __name__ == "__main__":

    thread_id = str(
        uuid7()
    )

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content":
                        "查询最近一个月订单金额超过1000元的客户"
                }
            ]
        },
        config
    )

    for message in result["messages"]:
        if hasattr(
                message,
                "pretty_print"
        ):
            message.pretty_print()
        else:
            print(
                f"{message.type}: "
                f"{message.content}"
            )