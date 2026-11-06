"""The agent's calculator: evaluate an arithmetic expression exactly and safely.

The LLM writes the expression ("(4506 / 14189) * 100"); this code computes it, because language models make
arithmetic mistakes. Python's eval() would also run any code the model wrote, so instead the expression is parsed
into a syntax tree (ast) and only these parts are evaluated:
  numbers, + - * / ** and unary minus, parentheses, and the functions avg, min, max, abs, round
Anything else (names, attribute access, calls to other functions, strings) is rejected with an error message.
"""

import ast
import operator

OPERATORS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
             ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos}
FUNCTIONS = {"avg": lambda *xs: sum(xs) / len(xs), "min": min, "max": max, "abs": abs, "round": round}
MAX_EXPONENT = 100  # 10 ** 10 ** 10 would hang the process


def _evaluate(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):  # type(), not isinstance: rejects True
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise ValueError(f"exponent {right} is too large")
        return OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in OPERATORS:
        return OPERATORS[type(node.op)](_evaluate(node.operand))
    if isinstance(node, ast.Tuple):
        raise ValueError("write numbers without thousands separators (1577, not 1,577)")
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FUNCTIONS
            and not node.keywords):
        return FUNCTIONS[node.func.id](*[_evaluate(arg) for arg in node.args])
    raise ValueError(f"not allowed in an expression: {ast.unparse(node)!r}")


def calculate(expression: str) -> dict:
    """{"expression", "value"} or {"expression", "error"}.

    Numbers must be written without thousands separators: in "avg(100,200)" a comma can't be told apart from
    one inside a number, so "1,577" is rejected with a message rather than guessed."""
    try:
        value = _evaluate(ast.parse(expression, mode="eval"))
        return {"expression": expression, "value": float(value)}
    except ZeroDivisionError:
        return {"expression": expression, "error": "division by zero"}
    except (SyntaxError, ValueError, TypeError) as e:
        return {"expression": expression, "error": str(e)}


if __name__ == "__main__":
    for expression in ["(4506 / 14189) * 100", "avg(1.2, 1.8, 2.7)", "1,577 + 2,000", "avg(100,200)",
                       "365 * avg(100, 120) / (900 + (130 - 110))", "-3.7 * 2", "1 / 0",
                       "__import__('os').system('ls')", "2 ** 10 ** 10", "True + 1"]:
        print(f"{expression!r:45} -> {calculate(expression)}")
