"""
多工具研究型 AI Agent (Multi-Tool Research Agent)
====================================================
一个进阶版 AI Agent：面对需要"查资料 + 计算 + 记录"的复杂问题，
它能自主判断该用哪个工具、按什么顺序使用，一步步推进到最终答案。

相比单工具 Agent，本项目的核心进阶点是【多工具编排 + 自主工具选择】：
Agent 拥有 3 个工具，每一步自己决定调用哪个——这正是智能体决策能力的体现。

三个工具：
  1. web_search    —— 联网搜索信息（用 DuckDuckGo，无需 API key）
  2. calculator    —— 安全地进行数学计算
  3. save_note     —— 把结论保存到本地文件

采用 ReAct 模式（Reasoning + Acting）：模型在"思考→行动→观察"的循环中推进。

运行方式：
    python agent.py "特斯拉和比亚迪2024年营收差多少？"
"""

import os
import sys
import json
import ast
import operator
import urllib.parse
import urllib.request
from openai import OpenAI

# 默认用 OpenRouter 上免费的模型；也可用环境变量 AGENT_MODEL 覆盖
# （OpenRouter 免费模型名单变动较快，如某个模型失效，换一个 :free 模型即可）
MODEL = os.environ.get("AGENT_MODEL", "openai/gpt-oss-20b:free")


# ============================================================
# 工具 1：网页搜索（DuckDuckGo Instant Answer API，免费无需 key）
# ============================================================
def web_search(query: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0 (research-agent)"}

    # 方案一：DuckDuckGo 即时答案 API
    try:
        url = "https://api.duckduckgo.com/?" + urllib.parse.urlencode({
            "q": query, "format": "json", "no_html": 1, "skip_disambig": 1,
        })
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
        if data.get("AbstractText"):
            return f"{data['AbstractText']}（来源：{data.get('AbstractSource','')}）"
        topics = data.get("RelatedTopics", [])
        snippets = [t["Text"] for t in topics if isinstance(t, dict) and t.get("Text")]
        if snippets:
            return " | ".join(snippets[:3])
    except Exception:
        pass  # 失败则尝试备用方案

    # 方案二：维基百科摘要 API（作为备用搜索源）
    try:
        title = urllib.parse.quote(query.replace(" ", "_"))
        url = f"https://zh.wikipedia.org/api/rest_v1/page/summary/{title}"
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
        if data.get("extract"):
            return f"{data['extract']}（来源：维基百科）"
    except Exception:
        pass

    return "未找到直接答案，建议换个更具体的关键词再搜一次。"


# ============================================================
# 工具 2：计算器（用 AST 安全求值，不用危险的 eval）
# ============================================================
_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Pow: operator.pow, ast.USub: operator.neg,
    ast.Mod: operator.mod,
}

def _safe_eval(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BinOp):
        return _OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp):
        return _OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("不支持的表达式")

def calculator(expression: str) -> str:
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree.body)
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算出错：{e}（只支持 + - * / ** % 等基本运算）"


# ============================================================
# 工具 3：保存笔记到本地文件
# ============================================================
def save_note(content: str) -> str:
    try:
        with open("research_notes.txt", "a", encoding="utf-8") as f:
            f.write(content + "\n" + "-" * 40 + "\n")
        return "已保存到 research_notes.txt"
    except Exception as e:
        return f"保存出错：{e}"


# ============================================================
# 工具注册表 + 给大模型的工具描述
# ============================================================
TOOL_FUNCS = {
    "web_search": web_search,
    "calculator": calculator,
    "save_note": save_note,
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "联网搜索信息。当你需要查找事实、数据、最新信息时使用。",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "搜索关键词"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "进行数学计算。支持 + - * / ** % 运算。当你需要算数时使用。",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string", "description": "数学表达式，如 (1000-800)*7.2"}},
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_note",
            "description": "把重要结论保存到本地文件。当你得出最终结论时使用。",
            "parameters": {
                "type": "object",
                "properties": {"content": {"type": "string", "description": "要保存的内容"}},
                "required": ["content"],
            },
        },
    },
]


# ============================================================
# Agent 主循环（ReAct：思考 → 行动 → 观察 → 重复）
# ============================================================
def run_agent(question: str, max_steps: int = 8):
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY"),
    )

    messages = [
        {
            "role": "system",
            "content": (
                "你是一个研究助手 Agent。你有三个工具：web_search（搜索）、"
                "calculator（计算）、save_note（保存结论）。"
                "请一步步思考，自主决定使用哪个工具来回答用户的问题。"
                "需要事实就搜索，需要算数就用计算器，得出最终结论后用 save_note 保存。"
            ),
        },
        {"role": "user", "content": question},
    ]

    print(f"\n问题：{question}\n" + "=" * 55)

    for step in range(max_steps):
        response = client.chat.completions.create(
            model=MODEL, messages=messages, tools=TOOLS,
        )
        msg = response.choices[0].message
        messages.append(msg)

        # 打印 Agent 的思考
        if msg.content and msg.content.strip():
            print(f"\n[第{step+1}步·思考] {msg.content.strip()}")

        if msg.tool_calls:
            for tc in msg.tool_calls:
                name = tc.function.name
                # 免费模型有时生成不规范的 JSON，做容错处理
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    # 尝试简单修复：去掉可能的多余字符后重试
                    raw = tc.function.arguments.strip()
                    try:
                        args = json.loads(raw.replace("\n", " "))
                    except json.JSONDecodeError:
                        err = f"工具参数解析失败，原始内容：{tc.function.arguments[:200]}"
                        print(f"\n[调用工具] {name}（参数解析失败，跳过）")
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": err + "。请重新生成格式正确的参数。",
                        })
                        continue
                print(f"\n[调用工具] {name}({args})")
                result = TOOL_FUNCS[name](**args)
                print(f"[工具结果] {result}")
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })
        else:
            # 没有再调用工具 = 给出最终答案
            print(f"\n[最终回答]\n{msg.content}\n" + "=" * 55)
            return

    print("\n（已达最大步数，结束）")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('用法：python agent.py "<你的问题>"')
        sys.exit(1)
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("错误：未设置 OPENROUTER_API_KEY 环境变量。")
        print("请先运行：export OPENROUTER_API_KEY='你的key'")
        sys.exit(1)

    run_agent(sys.argv[1])
