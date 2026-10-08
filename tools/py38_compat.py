"""Build Python 3.8-compatible artifacts from modern source trees."""

from __future__ import annotations

import argparse
import ast
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path
from typing import cast

IGNORED_DIR_NAMES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "out",
}
TYPING_NAME_MAP = {
    "defaultdict": "DefaultDict",
    "deque": "Deque",
    "dict": "Dict",
    "frozenset": "FrozenSet",
    "list": "List",
    "set": "Set",
    "tuple": "Tuple",
    "type": "Type",
}


def _is_docstring(statement: ast.stmt) -> bool:
    return (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Constant)
        and isinstance(statement.value.value, str)
    )


def _is_none_literal(expression: ast.expr) -> bool:
    return (
        isinstance(expression, ast.Constant) and expression.value is None
    ) or (isinstance(expression, ast.Name) and expression.id == "None")


def _looks_like_type_expression(expression: ast.expr) -> bool:
    if isinstance(
        expression,
        (ast.Attribute, ast.Constant, ast.Name, ast.Subscript),
    ):
        return True
    if isinstance(expression, ast.Tuple):
        return all(
            _looks_like_type_expression(item) for item in expression.elts
        )
    if isinstance(expression, ast.BinOp) and isinstance(
        expression.op,
        ast.BitOr,
    ):
        return _looks_like_type_expression(
            expression.left,
        ) and _looks_like_type_expression(expression.right)
    return False


class _TypeExpressionRewriter(ast.NodeTransformer):
    def __init__(self, typing_names: set[str]) -> None:
        self._typing_names = typing_names

    def _typing_reference(self, name: str) -> ast.Name:
        self._typing_names.add(name)
        return ast.Name(id=name, ctx=ast.Load())

    def _typing_subscript(
        self,
        name: str,
        expression: ast.expr,
    ) -> ast.Subscript:
        return ast.Subscript(
            value=self._typing_reference(name),
            slice=expression,
            ctx=ast.Load(),
        )

    def _flatten_union_operands(self, expression: ast.expr) -> list[ast.expr]:
        if (
            isinstance(expression, ast.Subscript)
            and isinstance(expression.value, ast.Name)
            and expression.value.id == "Union"
        ):
            if isinstance(expression.slice, ast.Tuple):
                return list(expression.slice.elts)
            return [expression.slice]
        return [expression]

    def visit_BinOp(self, node: ast.BinOp) -> ast.expr:
        if not isinstance(node.op, ast.BitOr):
            return cast("ast.expr", self.generic_visit(node))
        left = self.visit(node.left)
        right = self.visit(node.right)
        if _is_none_literal(left):
            return self._typing_subscript("Optional", right)
        if _is_none_literal(right):
            return self._typing_subscript("Optional", left)
        union_operands = self._flatten_union_operands(left)
        union_operands.extend(self._flatten_union_operands(right))
        return self._typing_subscript(
            "Union",
            ast.Tuple(elts=union_operands, ctx=ast.Load()),
        )

    def visit_Subscript(self, node: ast.Subscript) -> ast.Subscript:
        node = cast("ast.Subscript", self.generic_visit(node))
        if isinstance(node.value, ast.Name):
            replacement = TYPING_NAME_MAP.get(node.value.id)
            if replacement is not None:
                node.value = self._typing_reference(replacement)
        return node


class _ModuleTransformer(ast.NodeTransformer):
    def __init__(self) -> None:
        self._typing_names: set[str] = set()

    def _rewrite_type_expression(
        self,
        expression: ast.expr | None,
    ) -> ast.expr | None:
        if expression is None:
            return None
        rewriter = _TypeExpressionRewriter(self._typing_names)
        return rewriter.visit(expression)

    def _rewrite_decorator(self, decorator: ast.expr) -> ast.expr:
        if isinstance(decorator, ast.Call):
            decorator.func = self.visit(decorator.func)
            decorator.args = [self.visit(arg) for arg in decorator.args]
            decorator.keywords = [
                ast.keyword(arg=keyword.arg, value=self.visit(keyword.value))
                for keyword in decorator.keywords
                if keyword.arg != "slots"
            ]
            return decorator
        return self.visit(decorator)

    def _visit_block(self, statements: list[ast.stmt]) -> list[ast.stmt]:
        rewritten: list[ast.stmt] = []
        for statement in statements:
            transformed = self.visit(statement)
            if transformed is None:
                continue
            if isinstance(transformed, list):
                rewritten.extend(transformed)
            else:
                rewritten.append(transformed)
        return rewritten

    def _visit_arguments(self, arguments: ast.arguments) -> ast.arguments:
        for argument in arguments.posonlyargs:
            argument.annotation = self._rewrite_type_expression(
                argument.annotation,
            )
        for argument in arguments.args:
            argument.annotation = self._rewrite_type_expression(
                argument.annotation,
            )
        if arguments.vararg is not None:
            arguments.vararg.annotation = self._rewrite_type_expression(
                arguments.vararg.annotation,
            )
        for argument in arguments.kwonlyargs:
            argument.annotation = self._rewrite_type_expression(
                argument.annotation,
            )
        if arguments.kwarg is not None:
            arguments.kwarg.annotation = self._rewrite_type_expression(
                arguments.kwarg.annotation,
            )
        return arguments

    def _ensure_future_import(self, body: list[ast.stmt]) -> list[ast.stmt]:
        for statement in body:
            if (
                isinstance(statement, ast.ImportFrom)
                and statement.module == "__future__"
            ):
                if any(
                    alias.name == "annotations" for alias in statement.names
                ):
                    return body
        insertion_index = 0
        if body and _is_docstring(body[0]):
            insertion_index = 1
        while insertion_index < len(body):
            statement = body[insertion_index]
            if (
                not isinstance(statement, ast.ImportFrom)
                or statement.module != "__future__"
            ):
                break
            insertion_index += 1
        future_import = ast.ImportFrom(
            module="__future__",
            names=[ast.alias(name="annotations")],
            level=0,
        )
        return (
            body[:insertion_index] + [future_import] + body[insertion_index:]
        )

    def _ensure_typing_import(self, body: list[ast.stmt]) -> list[ast.stmt]:
        if not self._typing_names:
            return body
        required_names = sorted(self._typing_names)
        for statement in body:
            if (
                not isinstance(statement, ast.ImportFrom)
                or statement.module != "typing"
            ):
                continue
            imported_names = {alias.name for alias in statement.names}
            missing_names = [
                name for name in required_names if name not in imported_names
            ]
            if missing_names:
                statement.names.extend(
                    ast.alias(name=name) for name in missing_names
                )
            return body
        insertion_index = 0
        if body and _is_docstring(body[0]):
            insertion_index = 1
        while insertion_index < len(body):
            statement = body[insertion_index]
            if (
                not isinstance(statement, ast.ImportFrom)
                or statement.module != "__future__"
            ):
                break
            insertion_index += 1
        typing_import = ast.ImportFrom(
            module="typing",
            names=[ast.alias(name=name) for name in required_names],
            level=0,
        )
        return (
            body[:insertion_index] + [typing_import] + body[insertion_index:]
        )

    def visit_Module(self, node: ast.Module) -> ast.Module:
        node.body = self._visit_block(node.body)
        node.body = self._ensure_future_import(node.body)
        node.body = self._ensure_typing_import(node.body)
        return node

    def visit_Assign(self, node: ast.Assign) -> ast.Assign:
        node = cast("ast.Assign", self.generic_visit(node))
        if not node.targets or not all(
            isinstance(target, ast.Name) for target in node.targets
        ):
            return node
        target_names = [
            target.id
            for target in node.targets
            if isinstance(target, ast.Name)
        ]
        if not any(name[:1].isupper() for name in target_names):
            return node
        if _looks_like_type_expression(node.value):
            node.value = cast(
                "ast.expr",
                self._rewrite_type_expression(node.value),
            )
        return node

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AnnAssign:
        node.target = self.visit(node.target)
        node.annotation = cast(
            "ast.expr",
            self._rewrite_type_expression(node.annotation),
        )
        if node.value is not None:
            node.value = self.visit(node.value)
        return node

    def visit_AsyncFunctionDef(
        self,
        node: ast.AsyncFunctionDef,
    ) -> ast.AsyncFunctionDef:
        node.decorator_list = [
            self._rewrite_decorator(decorator)
            for decorator in node.decorator_list
        ]
        node.args = self._visit_arguments(node.args)
        node.returns = self._rewrite_type_expression(node.returns)
        node.body = self._visit_block(node.body)
        return node

    def visit_ClassDef(
        self,
        node: ast.ClassDef,
    ) -> ast.ClassDef | list[ast.stmt]:
        node.decorator_list = [
            self._rewrite_decorator(decorator)
            for decorator in node.decorator_list
        ]
        node.bases = [self.visit(base) for base in node.bases]
        node.keywords = [
            ast.keyword(arg=keyword.arg, value=self.visit(keyword.value))
            for keyword in node.keywords
        ]
        node.body = self._visit_block(node.body)
        type_params = list(getattr(node, "type_params", ()))
        if not type_params:
            return node
        self._typing_names.update({"Generic", "TypeVar"})
        typevar_names: list[str] = []
        assignments: list[ast.stmt] = []
        for type_param in type_params:
            name = getattr(type_param, "name", None)
            if not isinstance(name, str) or not name:
                continue
            typevar_names.append(name)
            assignments.append(
                ast.Assign(
                    targets=[ast.Name(id=name, ctx=ast.Store())],
                    value=ast.Call(
                        func=ast.Name(id="TypeVar", ctx=ast.Load()),
                        args=[ast.Constant(value=name)],
                        keywords=[],
                    ),
                ),
            )
        if typevar_names:
            slice_expression: ast.expr
            if len(typevar_names) == 1:
                slice_expression = ast.Name(
                    id=typevar_names[0],
                    ctx=ast.Load(),
                )
            else:
                slice_expression = ast.Tuple(
                    elts=[
                        ast.Name(id=name, ctx=ast.Load())
                        for name in typevar_names
                    ],
                    ctx=ast.Load(),
                )
            node.bases.append(
                ast.Subscript(
                    value=ast.Name(id="Generic", ctx=ast.Load()),
                    slice=slice_expression,
                    ctx=ast.Load(),
                ),
            )
        node.type_params = []
        return assignments + [node]

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        node.decorator_list = [
            self._rewrite_decorator(decorator)
            for decorator in node.decorator_list
        ]
        node.args = self._visit_arguments(node.args)
        node.returns = self._rewrite_type_expression(node.returns)
        node.body = self._visit_block(node.body)
        return node


def _copy_repo(source_root: Path, output_root: Path) -> None:
    if output_root.exists():
        shutil.rmtree(output_root)
    shutil.copytree(
        source_root,
        output_root,
        ignore=shutil.ignore_patterns(*sorted(IGNORED_DIR_NAMES)),
    )


def _transform_file(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    shebang = source.splitlines()[0] if source.startswith("#!") else ""
    tree = ast.parse(source)
    transformer = _ModuleTransformer()
    tree = transformer.visit(tree)
    ast.fix_missing_locations(tree)
    rendered = ast.unparse(tree)
    if shebang:
        rendered = f"{shebang}\n\n{rendered}"
    path.write_text(f"{rendered}\n", encoding="utf-8")


def _transform_repo(output_root: Path) -> None:
    for path in output_root.rglob("*.py"):
        _transform_file(path)


def _build_artifacts(output_root: Path, dist_dir: Path) -> None:
    if importlib.util.find_spec("build") is None:
        message = (
            "The 'build' package is required for py38 compatibility builds. "
            "Install it with 'python -m pip install build'."
        )
        raise RuntimeError(message)
    dist_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--sdist",
            "--wheel",
            "--outdir",
            str(dist_dir),
            str(output_root),
        ],
        check=True,
        cwd=output_root / "tools",
    )


def _get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "transform"))
    parser.add_argument(
        "--output-root",
        default="out/py38-src",
        help="Directory that receives the transformed source tree.",
    )
    parser.add_argument(
        "--dist-dir",
        default="out/py38-dist",
        help="Directory that receives built compatibility artifacts.",
    )
    return parser


def main() -> int:
    """Transform the repository tree and optionally build compatibility artifacts."""
    parser = _get_parser()
    arguments = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    output_root = (repo_root / arguments.output_root).resolve()
    dist_dir = (repo_root / arguments.dist_dir).resolve()

    _copy_repo(repo_root, output_root)
    _transform_repo(output_root)

    if arguments.command == "build":
        _build_artifacts(output_root, dist_dir)
        sys.stdout.write(f"{dist_dir}\n")
    else:
        sys.stdout.write(f"{output_root}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
