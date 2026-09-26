#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""mini_interpreter.py —— 逐行解释执行的小型脚本工具（纯 Python 标准库，单文件）。

脚本语言语法（用缩进表示语句块，每级缩进固定 4 个空格）：

    set 变量名 = 表达式      变量赋值
    print 表达式            输出表达式的值
    if 条件表达式:          条件为真执行缩进块
        ...
    else:                   可选，条件为假执行缩进块
        ...
    loop 次数表达式:        将缩进块重复执行指定次数
        ...

表达式支持：整数 / 浮点 / 字符串字面量、变量引用、
算术运算 + - * / // %、比较运算 == != < <= > >=、
逻辑运算 and / or / not、括号。字符串可用 + 拼接、用 * 重复。
# 之后为注释。

用法：
    python3 mini_interpreter.py 脚本文件      执行指定脚本
    python3 mini_interpreter.py < 脚本文件    从标准输入读脚本
    python3 mini_interpreter.py --demo        运行内置样例（含错误定位样例）
"""

from __future__ import annotations

import ast
import re
import sys
from dataclasses import dataclass, field

# 执行步数上限：每执行一条语句计 1 步。
# 本语言的 loop 次数在进入循环时求值并固定，本身不会无限循环，
# 但次数表达式可能来自变量（如 loop n: 且 n 被算成天文数字），
# 嵌套循环也会让步数爆炸。100_000 步对正常脚本绰绰有余，
# 而失控脚本会在亚秒级内被中止，因此取该值作为防死循环保险。
MAX_STEPS = 100_000

INDENT_UNIT = 4  # 每级缩进的空格数


class ScriptError(Exception):
    """脚本错误（词法/语法/运行时），携带行号用于定位。"""

    def __init__(self, line: int, message: str):
        self.line = line
        self.message = message
        super().__init__(f"第 {line} 行: {message}")


@dataclass
class Stmt:
    """一条语句的语法树节点。"""

    kind: str  # 'set' | 'print' | 'if' | 'loop'
    line: int  # 源脚本中的行号（从 1 开始）
    source: str  # 语句原文（用于跟踪输出）
    name: str = ""  # set 的变量名
    expr: str = ""  # 表达式文本（赋值右值 / 条件 / 循环次数）
    body: list = field(default_factory=list)  # if/loop 的语句块
    orelse: list = field(default_factory=list)  # else 的语句块


# ---------------------------------------------------------------- 词法辅助

_SET_RE = re.compile(r"^set\s+([A-Za-z_]\w*)\s*=\s*(.+)$")


def _strip_comment(line: str) -> str:
    """去掉行内 # 注释（字符串字面量内的 # 不算注释）。"""
    quote = None
    for i, ch in enumerate(line):
        if quote:
            if ch == quote:
                quote = None
        elif ch in ("'", '"'):
            quote = ch
        elif ch == "#":
            return line[:i]
    return line


def _check_parens(text: str, line: int) -> None:
    """显式检查括号配对（跳过字符串字面量），给出中文错误信息。"""
    quote = None
    stack = []
    pairs = {")": "(", "]": "[", "}": "{"}
    for ch in text:
        if quote:
            if ch == quote:
                quote = None
        elif ch in ("'", '"'):
            quote = ch
        elif ch in "([{":
            stack.append(ch)
        elif ch in ")]}":
            if not stack or stack[-1] != pairs[ch]:
                raise ScriptError(line, f"括号不配对: {text!r}")
            stack.pop()
    if quote:
        raise ScriptError(line, f"字符串引号未闭合: {text!r}")
    if stack:
        raise ScriptError(line, f"括号不配对: {text!r}")


# ---------------------------------------------------------------- 语法分析


def parse_script(text: str) -> list:
    """把脚本文本解析成语句树；格式错误抛出带行号的 ScriptError。"""
    rows = []
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = _strip_comment(raw)
        if not line.strip():
            continue
        leading = line[: len(line) - len(line.lstrip())]
        if "\t" in leading:
            raise ScriptError(lineno, "缩进错乱：缩进中不允许使用 Tab，请用空格")
        indent = len(leading)
        if indent % INDENT_UNIT != 0:
            raise ScriptError(
                lineno, f"缩进错乱：缩进量 {indent} 不是 {INDENT_UNIT} 的倍数"
            )
        rows.append((lineno, indent, line.strip()))
    stmts, _ = _parse_block(rows, 0, 0)
    return stmts


def _parse_block(rows, pos, indent):
    """解析同一缩进层级的语句序列，返回 (语句列表, 下一行下标)。"""
    stmts = []
    while pos < len(rows):
        lineno, ind, text = rows[pos]
        if ind < indent:
            break  # 缩进回退，本块结束
        if ind > indent:
            raise ScriptError(
                lineno, "缩进错乱：此处不应出现额外缩进（前面缺少 if/loop/else 引导？）"
            )
        stmt, pos = _parse_stmt(rows, pos, indent)
        stmts.append(stmt)
    return stmts, pos


def _parse_stmt(rows, pos, indent):
    lineno, _, text = rows[pos]
    keyword = text.split(None, 1)[0]

    if keyword == "set":
        m = _SET_RE.match(text)
        if not m:
            raise ScriptError(
                lineno, f"赋值语句格式错误，应为 'set 变量名 = 表达式': {text!r}"
            )
        _check_parens(m.group(2), lineno)
        return Stmt("set", lineno, text, name=m.group(1), expr=m.group(2)), pos + 1

    if keyword == "print":
        expr = text[len("print"):].strip()
        if not expr:
            raise ScriptError(lineno, "print 语句缺少要输出的表达式")
        _check_parens(expr, lineno)
        return Stmt("print", lineno, text, expr=expr), pos + 1

    if keyword == "if":
        cond = text[len("if"):].strip()
        if not cond.endswith(":"):
            raise ScriptError(lineno, "if 语句缺少结尾冒号 ':'")
        cond = cond[:-1].strip()
        if not cond:
            raise ScriptError(lineno, "if 语句缺少条件表达式")
        _check_parens(cond, lineno)
        body, pos = _parse_block(rows, pos + 1, indent + INDENT_UNIT)
        if not body:
            raise ScriptError(lineno, "if 的语句块为空（冒号后应有缩进的语句）")
        orelse = []
        if pos < len(rows) and rows[pos][1] == indent and rows[pos][2].startswith("else"):
            else_lineno, _, else_text = rows[pos]
            if else_text != "else:":
                raise ScriptError(else_lineno, f"else 语句格式错误，应为 'else:': {else_text!r}")
            orelse, pos = _parse_block(rows, pos + 1, indent + INDENT_UNIT)
            if not orelse:
                raise ScriptError(else_lineno, "else 的语句块为空")
        return Stmt("if", lineno, text, expr=cond, body=body, orelse=orelse), pos

    if keyword == "else":
        raise ScriptError(lineno, "else 没有配对的 if（或缩进未与 if 对齐）")

    if keyword == "loop":
        rest = text[len("loop"):].strip()
        if not rest.endswith(":"):
            raise ScriptError(lineno, "loop 语句缺少结尾冒号 ':'")
        rest = rest[:-1].strip()
        if not rest:
            raise ScriptError(lineno, "loop 语句缺少循环次数表达式")
        _check_parens(rest, lineno)
        body, pos = _parse_block(rows, pos + 1, indent + INDENT_UNIT)
        if not body:
            raise ScriptError(lineno, "loop 的语句块为空")
        return Stmt("loop", lineno, text, expr=rest, body=body), pos

    raise ScriptError(
        lineno, f"无法识别的语句（缺少关键字 set/print/if/loop）: {text!r}"
    )


# ---------------------------------------------------------------- 表达式求值


def eval_expr(expr_text: str, env: dict, line: int):
    """用 ast 模块安全地解析并求值表达式（不调用 eval，无法执行任意代码）。"""
    try:
        tree = ast.parse(expr_text, mode="eval")
    except SyntaxError as exc:
        raise ScriptError(line, f"表达式语法错误: {expr_text!r}（{exc.msg}）") from None
    try:
        return _eval_node(tree.body, env, line)
    except ScriptError:
        raise
    except (TypeError, ZeroDivisionError, ArithmeticError) as exc:
        raise ScriptError(line, f"运行时错误: {exc}") from None


def _eval_node(node, env, line):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, str, bool)):
            return node.value
        raise ScriptError(line, f"不支持的字面量: {node.value!r}")

    if isinstance(node, ast.Name):
        if node.id not in env:
            raise ScriptError(line, f"变量 '{node.id}' 未定义")
        return env[node.id]

    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left, env, line)
        right = _eval_node(node.right, env, line)
        op_type = type(node.op)
        if op_type not in _BIN_OPS:
            raise ScriptError(line, "不支持的运算符")
        return _BIN_OPS[op_type](left, right)

    if isinstance(node, ast.UnaryOp):
        operand = _eval_node(node.operand, env, line)
        if isinstance(node.op, ast.USub):
            return -operand
        if isinstance(node.op, ast.UAdd):
            return +operand
        if isinstance(node.op, ast.Not):
            return not operand
        raise ScriptError(line, "不支持的一元运算符")

    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            result = True
            for value_node in node.values:
                result = _eval_node(value_node, env, line)
                if not result:
                    return result
            return result
        if isinstance(node.op, ast.Or):
            result = False
            for value_node in node.values:
                result = _eval_node(value_node, env, line)
                if result:
                    return result
            return result

    if isinstance(node, ast.Compare):
        left = _eval_node(node.left, env, line)
        for op, comparator in zip(node.ops, node.comparators):
            right = _eval_node(comparator, env, line)
            op_type = type(op)
            if op_type not in _CMP_OPS:
                raise ScriptError(line, "不支持的比较运算符")
            if not _CMP_OPS[op_type](left, right):
                return False
            left = right
        return True

    raise ScriptError(line, f"不支持的表达式结构: {type(node).__name__}")


_BIN_OPS = {
    ast.Add: lambda a, b: a + b,   # 数值加法 / 字符串拼接
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,  # 数值乘法 / 字符串重复
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
}

_CMP_OPS = {
    ast.Eq: lambda a, b: a == b,
    ast.NotEq: lambda a, b: a != b,
    ast.Lt: lambda a, b: a < b,
    ast.LtE: lambda a, b: a <= b,
    ast.Gt: lambda a, b: a > b,
    ast.GtE: lambda a, b: a >= b,
}


# ---------------------------------------------------------------- 解释执行


class Interpreter:
    """逐条执行语句树，记录执行跟踪、程序输出与变量快照。"""

    def __init__(self):
        self.env = {}       # 变量表
        self.steps = 0      # 已执行步数
        self.records = []   # 跟踪与输出记录

    def _trace(self, stmt: Stmt):
        self.steps += 1
        if self.steps > MAX_STEPS:
            raise ScriptError(
                stmt.line, f"执行步数超过上限 {MAX_STEPS}，疑似死循环，已中止"
            )
        self.records.append(
            f"[跟踪] 步骤 {self.steps:>3} | 第 {stmt.line:>2} 行 | {stmt.source}"
        )

    def _snapshot(self):
        if self.env:
            pairs = ", ".join(f"{k} = {v!r}" for k, v in sorted(self.env.items()))
        else:
            pairs = "(暂无变量)"
        self.records.append(f"        变量 => {pairs}")

    def run(self, stmts):
        for stmt in stmts:
            self._exec(stmt)

    def _exec(self, st: Stmt):
        self._trace(st)

        if st.kind == "set":
            self.env[st.name] = eval_expr(st.expr, self.env, st.line)
            self._snapshot()

        elif st.kind == "print":
            value = eval_expr(st.expr, self.env, st.line)
            self.records.append(f"[输出] {value}")
            self._snapshot()

        elif st.kind == "if":
            cond = eval_expr(st.expr, self.env, st.line)
            self._snapshot()
            self.records.append(
                f"        条件 ({st.expr}) => {'真，走 if 块' if cond else '假，走 else 块'}"
            )
            self.run(st.body if cond else st.orelse)

        elif st.kind == "loop":
            count = eval_expr(st.expr, self.env, st.line)
            if isinstance(count, bool) or not isinstance(count, int):
                raise ScriptError(st.line, f"loop 次数必须是整数，得到 {count!r}")
            if count < 0:
                raise ScriptError(st.line, f"loop 次数不能为负数: {count}")
            self._snapshot()
            for i in range(count):
                self.records.append(f"        --- 第 {i + 1}/{count} 轮 ---")
                self.run(st.body)


def run_script(source: str):
    """执行脚本，返回 (跟踪记录列表, 错误对象或 None)。出错时记录保留到出错点。"""
    interp = Interpreter()
    try:
        stmts = parse_script(source)
        interp.run(stmts)
        return interp.records, None
    except ScriptError as exc:
        return interp.records, exc


# ---------------------------------------------------------------- 样例

SAMPLE_OK = """\
# 样例：赋值、数值与文本运算、比较、分支、循环
set name = "世界"
print "你好, " + name
set total = 0
loop 3:
    set total = total + 2
print total
if total >= 5:
    print "total 不小于 5"
else:
    print "total 小于 5"
set score = total * 10 - 4
if score % 2 == 0 and score > 50:
    print "score 是大于 50 的偶数"
else:
    print "score 不满足条件"
"""

ERROR_SAMPLES = [
    ("引用未定义变量", "set a = 1\nprint b\n"),
    ("缺少等号（语句格式错误）", "set c 3\n"),
    ("if 缺少冒号", "set y = 2\nif y > 1\n    print y\n"),
    ("括号不配对", "print (1 + 2\n"),
    ("缩进错乱", "set a = 1\n  print a\n"),
    ("步数超限（死循环防护）", "set a = 0\nloop 1000000:\n    set a = a + 1\n"),
]


def demo():
    print("=" * 64)
    print("【样例脚本】")
    print(SAMPLE_OK)
    print("=" * 64)
    print("【执行跟踪与结果】")
    records, err = run_script(SAMPLE_OK)
    print("\n".join(records))
    print("-" * 64)
    print("执行结束。" if err is None else f"[错误] {err}")

    for title, src in ERROR_SAMPLES:
        print()
        print("=" * 64)
        print(f"【错误样例：{title}】")
        print(src)
        print("-" * 64)
        records, err = run_script(src)
        if records:
            print("\n".join(records))
        print(f"[错误] {err}" if err else "执行结束。")


def main(argv):
    if len(argv) >= 2 and argv[1] == "--demo":
        demo()
        return 0
    if len(argv) >= 2:
        with open(argv[1], encoding="utf-8") as f:
            source = f.read()
    else:
        source = sys.stdin.read()
    records, err = run_script(source)
    if records:
        print("\n".join(records))
    if err:
        print(f"[错误] {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
