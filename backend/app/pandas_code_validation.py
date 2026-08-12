from __future__ import annotations

import ast


class CodeValidationError(ValueError):
    pass


KNOWN_FRAMES = frozenset({"students", "courses", "enrollments", "grades"})
ALLOWED_LIBRARY_ROOTS = frozenset({"pd", "np"})
FORBIDDEN_NAMES = frozenset(
    {
        "open", "exec", "eval", "compile", "__import__", "input",
        "globals", "locals", "vars", "getattr", "setattr", "delattr",
        "os", "sys", "subprocess", "socket", "pathlib", "requests",
    }
)
FORBIDDEN_IO_ATTRIBUTES = frozenset(
    {
        "to_csv", "to_excel", "to_pickle", "to_parquet", "to_sql", "to_json", "to_clipboard",
        "to_hdf", "to_feather", "to_stata",
        # pandas/numpy file-I/O entry points reachable via the allowed `pd`/`np` roots that
        # do not follow the to_*/read_* naming convention:
        "io", "get_handle", "save", "load", "fromfile", "tofile", "genfromtxt", "loadtxt",
        "savetxt", "memmap",
    }
)
FORBIDDEN_MUTATING_ATTRIBUTES = frozenset(
    # DataFrame/Series methods that mutate their receiver in place with no `inplace=`
    # parameter to gate on - `inplace=True` is already rejected outright below, but these
    # need their own check since they have no such keyword to catch.
    {"pop", "update", "insert"}
)
FORBIDDEN_NODE_TYPES = (
    ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef,
    ast.ClassDef, ast.While, ast.With, ast.AsyncWith, ast.Try,
    ast.Raise, ast.Yield, ast.YieldFrom, ast.AsyncFor, ast.Global, ast.Nonlocal,
)
if hasattr(ast, "TryStar"):
    FORBIDDEN_NODE_TYPES = FORBIDDEN_NODE_TYPES + (ast.TryStar,)

_BUILTIN_ALLOWLIST = frozenset(
    {
        "len", "range", "min", "max", "sum", "sorted", "round", "abs",
        "int", "float", "str", "bool", "list", "dict", "set", "tuple",
        "enumerate", "zip", "map", "filter", "isinstance",
    }
)


def _is_dunder(name: str) -> bool:
    return name.startswith("__") and name.endswith("__")


def _root_name(node: ast.AST) -> str | None:
    while isinstance(node, (ast.Attribute, ast.Subscript)):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def validate_code(code: str, max_length: int = 4000) -> None:
    if len(code) > max_length:
        raise CodeValidationError(f"Generated code exceeds the maximum allowed length of {max_length} characters")
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as error:
        raise CodeValidationError(f"Generated code is not valid Python: {error}") from error

    assigned_names: set[str] = set()
    result_assigned = False

    for node in ast.walk(tree):
        if isinstance(node, FORBIDDEN_NODE_TYPES):
            raise CodeValidationError(f"Disallowed statement type: {type(node).__name__}")

        if isinstance(node, ast.Name):
            if node.id in FORBIDDEN_NAMES:
                raise CodeValidationError(f"Disallowed identifier: {node.id}")
            if _is_dunder(node.id):
                raise CodeValidationError(f"Disallowed dunder identifier: {node.id}")
            if isinstance(node.ctx, ast.Store):
                if node.id in KNOWN_FRAMES:
                    raise CodeValidationError(f"Reassignment of source dataframe '{node.id}' is not allowed")
                assigned_names.add(node.id)
                if node.id == "result":
                    result_assigned = True

        if isinstance(node, ast.Lambda):
            # lambda parameters are `ast.arg` nodes, not `ast.Name(ctx=Store)`, so they must be
            # collected separately - pandas `.agg(col=('x', lambda values: ...))` is a common,
            # legitimate pattern and must not be rejected as an unknown name.
            all_args = list(node.args.args) + list(node.args.posonlyargs) + list(node.args.kwonlyargs)
            if node.args.vararg:
                all_args.append(node.args.vararg)
            if node.args.kwarg:
                all_args.append(node.args.kwarg)
            assigned_names.update(arg.arg for arg in all_args)

        if isinstance(node, ast.Attribute):
            if _is_dunder(node.attr):
                raise CodeValidationError(f"Disallowed dunder attribute: {node.attr}")
            if node.attr in FORBIDDEN_IO_ATTRIBUTES or node.attr.startswith("read_"):
                raise CodeValidationError(f"Disallowed I/O method: {node.attr}")
            if node.attr in FORBIDDEN_MUTATING_ATTRIBUTES:
                raise CodeValidationError(f"Disallowed in-place-mutating method: {node.attr}")

        if isinstance(node, ast.keyword):
            # Reject `inplace=` outright, for ANY value - not just a literal `True`. Generated
            # code never legitimately needs `inplace=` at all (the contract is "assign the
            # result to `result`"), and checking only `ast.Constant(True)` is trivially bypassed
            # by a variable (`flag = True; df.drop(inplace=flag)`) or an expression
            # (`inplace=(1 == 1)`) - a blanket reject closes that off structurally rather than
            # trying to prove an arbitrary expression is never truthy.
            if node.arg == "inplace":
                raise CodeValidationError("The `inplace` keyword argument is not allowed")
            # `**kwargs`-style call unpacking has `arg is None`; block it outright so a keyword
            # like `inplace=True` can never be smuggled past the check above inside a dict
            # (`enrollments.drop(**{'inplace': True})`).
            if node.arg is None:
                raise CodeValidationError("Keyword-argument unpacking (**kwargs) is not allowed in a call")

        if isinstance(node, (ast.Subscript, ast.Attribute)) and isinstance(getattr(node, "ctx", None), (ast.Store, ast.Del)):
            root = _root_name(node)
            if root in KNOWN_FRAMES:
                raise CodeValidationError(f"Mutation of source dataframe '{root}' is not allowed")

    if not result_assigned:
        raise CodeValidationError("Generated code must assign the final answer to a variable named `result`")

    safe_roots = KNOWN_FRAMES | ALLOWED_LIBRARY_ROOTS | assigned_names | _BUILTIN_ALLOWLIST
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id not in safe_roots:
                raise CodeValidationError(f"Reference to an unknown or disallowed name: {node.id}")
