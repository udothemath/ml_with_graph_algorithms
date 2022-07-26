"""User-defined function parser."""
import ast
from typing import List


class UDFParser:
    """User-defined function parser.

    This parser aims at parsing user-defined `run_all` function and
    construct the corresponding numba kernel.

    Parameters:
        udf_logic_cls_path: file path of user-defined logic class
    """

    def __init__(self, udf_logic_cls_path: str):
        self.udf_logic_cls_path = udf_logic_cls_path

    def parse_run_all(self) -> ast.FunctionDef:
        """Parse user-defined `run_all` and construct corresponding
        numba kernel.

        Return:
            run_all_numba: `run_all` numba kernel
        """
        # Get user-defined logic class and function nodes
        udf_logic_cls_node = self.__get_udf_logic_cls_node()
        udf_run_all_node = self.__get_udf_run_all_node(udf_logic_cls_node)

        # Get user-defined `run_all` input columns (arguments)
        run_all_inputs = udf_run_all_node.args.args
        run_all_inputs.pop(0)  # Remove parameter `self`

        # Get user-defined `run_all` output columns (returns)
        run_all_outputs = self.__get_udf_run_all_outputs(udf_run_all_node)

        # Convert class method call to normal function call
        cls_method_trafo = ClassMethodTransformer()
        run_all_inner = cls_method_trafo.visit(udf_run_all_node)

        # Build inner functions within numba kernel from the class methods used in run_all
        inner_fns = self.__build_inner_sub_logics(udf_logic_cls_node, cls_method_trafo.sub_logics_)
        inner_fns.append(run_all_inner)

        # Construct `run_all` numba kernel
        run_all_numba = self.__gen_run_all_numba_kernel(run_all_inputs, run_all_outputs, inner_fns)

        return run_all_numba

    def __get_udf_logic_cls_node(self) -> ast.ClassDef:
        """Return user-defined logic class node.

        Return:
            udf_logic_cls_node: user-defined logic class node
        """
        with open(self.udf_logic_cls_path, "r") as f:
            udf_logic_module = ast.parse(f.read())

        for node in udf_logic_module.body:
            if isinstance(node, ast.ClassDef) and node.name.endswith("Logic"):
                udf_logic_cls_node = node

        return udf_logic_cls_node

    def __get_udf_run_all_node(self, udf_logic_cls_node: ast.ClassDef) -> ast.FunctionDef:
        """Return user-defined `run_all` function node.

        Parameters:
            udf_logic_cls_node: user-defined logic class node

        Return:
            udf_run_all_node: user-defined `run_all` function node
        """
        for node in udf_logic_cls_node.body:
            if isinstance(node, ast.FunctionDef) and node.name == "run_all":
                udf_run_all_node = node

        return udf_run_all_node

    def __get_udf_run_all_outputs(self, udf_run_all_node: ast.FunctionDef) -> List[ast.Name]:
        """Return output columns of user-defined `run_all` function.

        Parameters:
            udf_run_all_node: user-defined `run_all` function node

        Return:
            run_all_outputs: list of output columns
        """
        for node in udf_run_all_node.body:
            if isinstance(node, ast.Return):
                run_all_outputs = node.value.elts

        return run_all_outputs

    def __build_inner_sub_logics(
        self, udf_logic_cls_node: ast.ClassDef, sub_logics: List[str]
    ) -> List[ast.FunctionDef]:
        """Return inner functions called in user-defined `run_all`.

        Parameters:
            udf_logic_cls_node: user-defined logic class node
            sub_logics: sub-logics called in user-defined `run_all`

        Return:
            inner_fns: list of inner functions
        """
        inner_fns = []
        for node in udf_logic_cls_node.body:
            if isinstance(node, ast.FunctionDef) and node.name in sub_logics:
                node.args.args.pop(0)  # Remove parameter `self`
                inner_fns.append(node)

        return inner_fns

    def __gen_run_all_numba_kernel(
        self, input_cols: List[ast.arg], output_cols: List[ast.Name], inner_fns: List[ast.FunctionDef]
    ) -> ast.FunctionDef:
        """Return numba kernel function corresponding to user-defined
        `run_all` function.

        Parameters:
            input_cols: list of input columns
            output_cols: list of output columns
            inner_fns: list of inner functions

        Return:
            run_all_numba: numba kernel function
        """
        # Reconstruct numba kernel function arguments
        run_all_numba_args = ast.arguments(
            posonlyargs=[],
            args=input_cols + output_cols,  # type: ignore
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[],
        )

        # Reconstruct numba kernel function body
        run_all_numba_body = inner_fns.copy()
        inner_for = self.__build_inner_for(input_cols, output_cols)
        run_all_numba_body.append(inner_for)  # type: ignore

        # Define numba kernal function
        run_all_numba = ast.FunctionDef(
            name="run_all_numba",
            args=run_all_numba_args,
            body=run_all_numba_body,
            decorator_list=[],
            returns=None,
            type_comment=None,
        )

        return run_all_numba

    def __build_inner_for(self, input_cols: List[ast.arg], output_cols: List[ast.Name]) -> ast.For:
        """Build numba kernel inner for loop.

        Parameters:
            input_cols: list of input columns
            output_cols: list of output columns

        Return:
            inner_for: numba kernel inner for loop
        """
        inner_for = ast.For(
            target=self.__build_inner_for_target(input_cols),
            iter=self.__build_inner_for_iter(input_cols),
            body=self.__build_inner_for_body(input_cols, output_cols),
            orelse=[],
            type_comment=False,
        )

        return inner_for

    def __build_inner_for_target(self, input_cols: List[ast.arg]) -> ast.Tuple:
        """Build `target` field of numba kernel inner for loop.

        Parameters:
            input_cols: list of input columns

        Return:
            inner_for_target: `target` field of numba kernel inner for
                loop
        """
        inner_for_target = ast.Tuple(
            elts=[
                ast.Name(id="i", ctx=ast.Store()),
                ast.Tuple(
                    elts=[ast.Name(id=f"x{i}", ctx=ast.Store()) for i in range(len(input_cols))],
                    ctx=ast.Store(),
                ),
            ],
            ctx=ast.Store(),
        )

        return inner_for_target

    def __build_inner_for_iter(self, input_cols: List[ast.arg]) -> ast.Call:
        """Build `iter` field of numba kernel inner for loop.

        Parameters:
            input_cols: list of input columns

        Return:
            inner_for_iter: `iter` field of numba kernel inner for
                loop
        """
        inner_for_iter = ast.Call(
            func=ast.Name(
                id="enumerate",
                ctx=ast.Load(),
            ),
            args=[
                ast.Call(
                    func=ast.Name(
                        id="zip",
                        ctx=ast.Load(),
                    ),
                    args=input_cols,
                    keywords=[],
                ),
            ],
            keywords=[],
        )

        return inner_for_iter

    def __build_inner_for_body(self, input_cols: List[ast.arg], output_cols: List[ast.Name]) -> ast.Assign:
        """Build `body` field of numba kernel inner for loop.

        Parameters:
            input_cols: list of input columns
            output_cols: list of output columns

        Return:
            inner_for_body: `body` field of numba kernel inner for
                loop
        """
        inner_for_body = ast.Assign(
            targets=[
                ast.Tuple(
                    elts=[
                        ast.Subscript(
                            value=output,
                            slice=ast.Index(value=ast.Name(id="i", ctx=ast.Load())),
                            ctx=ast.Store(),
                        )
                        for output in output_cols
                    ],
                    ctx=ast.Store(),
                )
            ],
            value=ast.Call(
                func=ast.Name(id="run_all", ctx=ast.Load()),
                args=[ast.Name(id=f"x{i}", ctx=ast.Load()) for i in range(len(input_cols))],
                keywords=[],
            ),
            type_comment=False,
        )

        return inner_for_body


class ClassMethodTransformer(ast.NodeTransformer):
    """Node transformer converting class method call to normal function
    call.

    All class methods called in `run_all` are recorded for further
    inner function reconstruction.

    Attributes:
        sub_logics_: sub-logics called in user-defined `run_all`
    """

    sub_logics_: List[str]

    def __init__(self) -> None:
        self.sub_logics_ = []

    def visit_Call(self, node: ast.Call) -> ast.Call:
        """Apply transformer to visited node.

        Parameters:
            node: visited node to transform

        Return:
            node_trans: transformed node
        """
        if node.func.value.id == "self":
            sub_logic_name = node.func.attr
            if sub_logic_name not in self.sub_logics_:
                self.sub_logics_.append(sub_logic_name)

        node_trans = ast.copy_location(
            ast.Call(
                func=ast.Name(
                    id=node.func.attr,
                    ctx=ast.Load(),
                ),
                args=node.args,
                keywords=node.keywords,
            ),
            node,
        )

        return node_trans
