"""Where a call lands, what it is handed, and what comes back.

The walk crossed a function boundary in two ways only: a handover the map had
recorded, and a ``self.<field>`` answered from the class's other methods.  It
never followed a value into the function that returned it or out to the call
that passed it in, so every cross-function shape the corpus has was either a
rule of its own -- a literal returned one hop up, a mapping read out of a
zero-argument method, ``.add()`` on a bare name -- or a stop.  Python has no
end of those shapes, and each case found the next one.

This module is the one place a call is read, three ways:

* :meth:`Calls.callees` -- which function definitions a call can land in,
  read the way a type checker reads it: ``self.m`` through the class and its
  bases, ``obj.m`` through what ``obj`` is declared to be, a bare name through
  the enclosing function, the module and its imports, ``Class(...)`` as its
  ``__init__``.  Never by the method's name alone -- a crossing that matched
  callees by bare name gave 222 read sites the same two false leads -- so a
  call it cannot place comes back with the reason, not a guess.
* :meth:`Calls.bind` -- which expression at the call each parameter holds.
* :meth:`Calls.returns` -- what the function hands back, slot by slot.

What a method changes on its object is read by :meth:`Calls.writes`, so that
``self.result.set_fatal()`` is a write of ``status_code`` like an assignment.

Its parse cache is its own and spends none of the walk's budget, like the
declared-type reading it is built on.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import NamedTuple, Optional

from bamboo.codemap.panda import pathcond

#: Why a call could not be placed.  Each is a class of the census, so the
#: count of calls the walk could not enter says what would enter them.
UNPLACED_OUTSIDE = "outside the corpus"
UNPLACED_RECEIVER = "the receiver's class is not declared"
UNPLACED_FORWARDED = "the receiver forwards every call through __getattr__"
UNPLACED_NO_METHOD = "the declared class has no such method"
UNPLACED_DYNAMIC = "the callee is chosen at run time"
UNPLACED_NAME = "the name is bound to something that is not a function here"
#: Beside targets, not instead of them: the receiver is one of these classes
#: or another one some way of building it does not show.
UNPLACED_OPEN = "and possibly a class this reading cannot name"


#: Parsed modules shared by every reading of the same files.  A walk is made
#: once per arm, per statement and per printed line, and the corpus index
#: behind :meth:`Calls.callers` would otherwise be rebuilt for each.  Keyed by
#: the file's stat, so a tree edited under a reading is read afresh.
_PARSED: dict[tuple[str, int, int], Optional[ast.Module]] = {}
#: ``callers`` index and answers, per set of roots.
_INDEXES: dict[tuple, dict[str, list["CallSite"]]] = {}
_CALLERS: dict[tuple, list["CallSite"]] = {}


def _roots_key(roots: dict[str, Path]) -> tuple:
    return tuple(sorted((package, str(root)) for package, root in roots.items()))


class Target(NamedTuple):
    """One function definition a call can land in."""

    file: str
    func: ast.AST
    cls: Optional[ast.ClassDef]

    @property
    def owner(self) -> str:
        return f"{self.file}::{self.func.name}"  # type: ignore[attr-defined]


class Placed(NamedTuple):
    """Where a call lands: the definitions, or why none is named."""

    targets: tuple[Target, ...] = ()
    unplaced: str = ""


class CallSite(NamedTuple):
    """One call, in the function that makes it."""

    file: str
    func: ast.AST
    call: ast.Call

    @property
    def owner(self) -> str:
        return f"{self.file}::{self.func.name}"  # type: ignore[attr-defined]


def own_nodes(scope: ast.AST):
    """Every node of *scope*'s own body; a nested function or class runs elsewhere."""
    for child in ast.iter_child_nodes(scope):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            continue
        yield child
        yield from own_nodes(child)


def returns(func: ast.AST) -> list[ast.expr]:
    """The values *func* hands back, in source order; a bare ``return`` is ``None``."""
    return [
        node.value if node.value is not None else ast.Constant(value=None)
        for node in own_nodes(func)
        if isinstance(node, ast.Return)
    ]


def _is_static(func: ast.AST) -> bool:
    return any(
        isinstance(decorator, ast.Name) and decorator.id == "staticmethod"
        for decorator in getattr(func, "decorator_list", [])
    )


def bind(call: ast.Call, target: Target) -> dict[str, Optional[ast.expr]]:
    """``{parameter: the expression the call hands it}``; ``None`` where it is not said.

    Positional, keyword and default, in Python's order.  A ``*args`` or
    ``**kwargs`` at the call makes every parameter it could fill unknown
    rather than guessed; a parameter the call leaves to its default holds the
    default.  ``self`` is left out of a method reached through an instance.
    """
    args = target.func.args  # type: ignore[attr-defined]
    positional = [*args.posonlyargs, *args.args]
    if target.cls is not None and not _is_static(target.func) and positional:
        positional = positional[1:]
    defaults = dict(
        zip([p.arg for p in positional][len(positional) - len(args.defaults):], args.defaults, strict=True)
    )
    passed = list(call.args)
    spread = any(isinstance(one, ast.Starred) for one in passed)
    named = {keyword.arg: keyword.value for keyword in call.keywords if keyword.arg}
    spread_named = any(keyword.arg is None for keyword in call.keywords)
    bound: dict[str, Optional[ast.expr]] = {}
    for index, parameter in enumerate(positional):
        if not spread and index < len(passed) and not isinstance(passed[index], ast.Starred):
            bound[parameter.arg] = passed[index]
        elif parameter.arg in named:
            bound[parameter.arg] = named[parameter.arg]
        elif spread or spread_named:
            bound[parameter.arg] = None
        else:
            bound[parameter.arg] = defaults.get(parameter.arg)
    for parameter, default in zip(args.kwonlyargs, args.kw_defaults, strict=True):
        if parameter.arg in named:
            bound[parameter.arg] = named[parameter.arg]
        else:
            bound[parameter.arg] = None if spread_named else default
    return bound


def said(call: ast.Call, target: Target) -> set[str]:
    """The parameters *call* fills itself, by position or by keyword -- not by default.

    ``self``/``cls`` is not among them: the call names the object it is made
    on, which is a different thing from passing it.
    """
    args = target.func.args  # type: ignore[attr-defined]
    positional = [*args.posonlyargs, *args.args]
    if target.cls is not None and not _is_static(target.func) and positional:
        positional = positional[1:]
    filled = {
        parameter.arg
        for index, parameter in enumerate(positional)
        if index < len(call.args) and not any(isinstance(one, ast.Starred) for one in call.args[: index + 1])
    }
    filled |= {keyword.arg for keyword in call.keywords if keyword.arg}
    return filled


def receiver_parameter(target: Target) -> Optional[str]:
    """``self`` or ``cls`` -- the parameter a method call fills with the object it is made on."""
    args = target.func.args  # type: ignore[attr-defined]
    positional = [*args.posonlyargs, *args.args]
    if target.cls is None or _is_static(target.func) or not positional:
        return None
    return positional[0].arg


class Calls:
    """Calls read against one tree, with a corpus index built the first time it is asked."""

    def __init__(self, types: Types) -> None:
        self.types = types
        self._placed: dict[int, Placed] = {}
        self._by_name: Optional[dict[str, list[CallSite]]] = None
        self._callers: dict[tuple[str, int], list[CallSite]] = {}

    # -- where a call lands -------------------------------------------------

    def callees(self, file: str, func: ast.AST, call: ast.Call) -> Placed:
        key = id(call)
        if key not in self._placed:
            self._placed[key] = self._place(file, func, call)
        return self._placed[key]

    def _place(self, file: str, func: ast.AST, call: ast.Call) -> Placed:
        called = call.func
        if isinstance(called, ast.Name):
            return self._named(file, func, called.id)
        if not isinstance(called, ast.Attribute):
            return Placed((), UNPLACED_DYNAMIC)
        receiver = called.value
        if isinstance(receiver, (ast.Constant, ast.JoinedStr, ast.List, ast.Dict, ast.Set, ast.Tuple)):
            # ``",".join(...)``, ``f"...".format``: a builtin type's method.
            return Placed((), UNPLACED_OUTSIDE)
        if isinstance(receiver, ast.Name) and receiver.id in ("self", "cls"):
            cls = self.types.method_class(file, func)
            if cls is None:
                return Placed((), UNPLACED_RECEIVER)
            return self._method_of(cls, called.attr)
        if (
            isinstance(receiver, ast.Call)
            and isinstance(receiver.func, ast.Name)
            and receiver.func.id == "super"
        ):
            cls = self.types.method_class(file, func)
            if cls is None:
                return Placed((), UNPLACED_RECEIVER)
            lineage = self.types.lineage(*cls)[1:]
            for where, holder in lineage:
                found = self._declared_in(where, holder, called.attr)
                if found is not None:
                    return Placed((found,))
            return Placed((), UNPLACED_NO_METHOD)
        module = self.types.module_named(file, receiver)
        if module is not None:
            return self._module_member(module, called.attr)
        # ``JobBrokerBase.__init__(self, ...)``: a method read off its class.
        named = self.types.class_named(file, receiver)
        if named is not None:
            return self._method_of(named, called.attr)
        declared = self.types.of_expression(file, func, receiver)
        if declared is None:
            # Nothing declares it -- or it is declared ``Any`` -- so read what
            # it was built from.  Every class it can be is a target; where some
            # way of building it cannot be read, that is said beside them.
            classes, opened = self.types.candidates(file, func, receiver)
            targets: list[Target] = []
            for where, name in sorted(classes):
                cls = self.types.class_def(where, name)
                if cls is not None:
                    targets.extend(self._method_of((where, cls), called.attr).targets)
            if targets:
                return Placed(tuple(targets), UNPLACED_OPEN if opened else "")
            root = receiver
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name) and self.types.imported_from_outside(file, root.id):
                return Placed((), UNPLACED_OUTSIDE)
            return Placed((), UNPLACED_RECEIVER)
        return self._method_of(declared, called.attr)

    def _declared_in(self, file: str, cls: ast.ClassDef, name: str) -> Optional[Target]:
        for node in cls.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
                return Target(file, node, cls)
        return None

    def _method_of(self, declared: tuple[str, ast.ClassDef], name: str) -> Placed:
        file, cls = declared
        for where, holder in self.types.lineage(file, cls):
            found = self._declared_in(where, holder, name)
            if found is not None:
                return Placed((found,))
        for where, holder in self.types.lineage(file, cls):
            if self._declared_in(where, holder, "__getattr__") is not None:
                return Placed((), UNPLACED_FORWARDED)
        return Placed((), UNPLACED_NO_METHOD)

    def _named(self, file: str, func: ast.AST, name: str) -> Placed:
        # A function defined inside this one, or inside one that holds it.
        scope: Optional[ast.AST] = func
        while scope is not None:
            for node in own_nodes(scope):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
                    return Placed((Target(file, node, None),))
            for node in ast.iter_child_nodes(scope):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
                    return Placed((Target(file, node, None),))
            scope = self.types.enclosing_function(file, scope)
        tree = self.types.tree(file)
        if tree is None:
            return Placed((), UNPLACED_OUTSIDE)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
                return Placed((Target(file, node, None),))
        constructed = self.types.class_named(file, ast.Name(id=name, ctx=ast.Load()))
        if constructed is not None:
            init = self._method_of(constructed, "__init__")
            return init if init.targets else Placed((), "")
        held = self.types.imports(file).get(name)
        if held is not None:
            module, attribute = held
            target = self.types.module_file(module)
            if target is None:
                return Placed((), UNPLACED_OUTSIDE)
            return self._module_member(target, attribute or name)
        return Placed((), UNPLACED_NAME)

    def _module_member(self, file: str, name: str) -> Placed:
        tree = self.types.tree(file)
        if tree is None:
            return Placed((), UNPLACED_OUTSIDE)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
                return Placed((Target(file, node, None),))
            if isinstance(node, ast.ClassDef) and node.name == name:
                init = self._method_of((file, node), "__init__")
                return init if init.targets else Placed((), "")
        held = self.types.imports(file).get(name)
        if held is not None and held[0]:
            target = self.types.module_file(held[0])
            if target is not None and target != file:
                return self._module_member(target, held[1] or name)
        return Placed((), UNPLACED_NAME)

    # -- who calls a function ----------------------------------------------

    def _index(self) -> dict[str, list[CallSite]]:
        """Every call in the corpus, by the name it calls -- a candidate list only.

        A name is how to *find* the calls worth placing, never how to place
        one: each is resolved by :meth:`callees` before it is kept.
        """
        if self._by_name is not None:
            return self._by_name
        shared = _INDEXES.get(_roots_key(self.types.roots))
        if shared is not None:
            self._by_name = shared
            return shared
        found: dict[str, list[CallSite]] = {}
        for file in self.types.corpus_files():
            tree = self.types.tree(file)
            if tree is None:
                continue
            for func in ast.walk(tree):
                if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                for node in own_nodes(func):
                    if not isinstance(node, ast.Call):
                        continue
                    called = node.func
                    name = (
                        called.attr if isinstance(called, ast.Attribute)
                        else called.id if isinstance(called, ast.Name) else ""
                    )
                    if name:
                        found.setdefault(name, []).append(CallSite(file, func, node))
        self._by_name = found
        _INDEXES[_roots_key(self.types.roots)] = found
        return found

    def callers(self, target: Target) -> list[CallSite]:
        """The calls in the corpus that :meth:`callees` places at *target*."""
        key = (target.file, target.func.lineno)  # type: ignore[attr-defined]
        if key in self._callers:
            return self._callers[key]
        shared = (_roots_key(self.types.roots), *key)
        if shared in _CALLERS:
            self._callers[key] = _CALLERS[shared]
            return _CALLERS[shared]
        name = target.func.name  # type: ignore[attr-defined]
        if name == "__init__" and target.cls is not None:
            name = target.cls.name
        kept = [
            site
            for site in self._index().get(name, [])
            if any(
                (placed.file, placed.func.lineno) == key  # type: ignore[attr-defined]
                for placed in self.callees(site.file, site.func, site.call).targets
            )
        ]
        self._callers[key] = kept
        _CALLERS[shared] = kept
        return kept

    #: How many chains ``called_from`` lists before it counts the rest.
    CALLED_FROM = 16

    def called_from(self, target: Target) -> list[str]:
        """``file:line`` of each call placed at *target*, and one hop out where its caller is called.

        Sorted, capped at :data:`CALLED_FROM` with a closing ``… N more``.
        """
        chains: list[str] = []
        for site in self.callers(target):
            at = f"{site.file}:{site.call.lineno}"
            owned = self.types.method_class(site.file, site.func)
            outer = self.callers(Target(site.file, site.func, owned[1] if owned else None))
            if outer:
                chains.extend(f"{one.file}:{one.call.lineno} -> {at}" for one in outer)
            else:
                chains.append(at)
        chains = sorted(set(chains))
        if len(chains) > self.CALLED_FROM:
            return [*chains[: self.CALLED_FROM], f"… {len(chains) - self.CALLED_FROM} more"]
        return chains

    #: How many calls out a constant is looked for, and how many calls one
    #: function may have before its callers are not read one by one.
    CONSTANT_HOPS = 3
    CONSTANT_CALLERS = 16

    def handed_constants(
        self, target: Target, names: set[str], depth: int = 0
    ) -> list[tuple[tuple[str, ...], dict[str, object]]]:
        """Per call placed at *target*: where it came from, and the constants it hands *names*.

        ``(("pkg/tb.py:551", "pkg/cmd.py:143"), {"onlyActive": True})`` -- the
        call into *target* first.  A name the caller hands on from a parameter
        of its own that it never rebinds is looked for at that caller's
        callers, so ``JediTaskBuffer``'s forwarding method is read through to
        ``TaskCommando``.  A call handing none of *names* a constant is left
        out: it runs the function as the general reading already reads it.
        """
        sites = self.callers(target)
        if not sites or len(sites) > self.CONSTANT_CALLERS:
            return []
        found: list[tuple[tuple[str, ...], dict[str, object]]] = []
        for site in sites:
            bound = bind(site.call, target)
            at = f"{site.file}:{site.call.lineno}"
            values: dict[str, object] = {}
            forwarded: dict[str, str] = {}
            for name in names:
                handed = bound.get(name)
                if isinstance(handed, ast.Constant):
                    values[name] = handed.value
                elif (
                    isinstance(handed, ast.Name)
                    and handed.id in _parameter_names(site.func)
                    and not _rebinds(site.func, handed.id)
                ):
                    forwarded[name] = handed.id
            outer: list[tuple[tuple[str, ...], dict[str, object]]] = []
            if forwarded and depth + 1 < self.CONSTANT_HOPS:
                owned = self.types.method_class(site.file, site.func)
                caller = Target(site.file, site.func, owned[1] if owned else None)
                outer = self.handed_constants(caller, set(forwarded.values()), depth + 1)
            for via, more in outer:
                merged = dict(values)
                merged.update({name: more[held] for name, held in forwarded.items() if held in more})
                found.append(((at, *via), merged))
            if not outer and values:
                found.append(((at,), values))
        return found

    # -- what a method changes on its object --------------------------------

    def writes(self, target: Target, depth: int = 0) -> list[tuple[str, ast.expr, ast.AST, Target]]:
        """``(field, value, site, where)`` for each write *target* makes to its own object.

        An assignment to ``self.<field>`` (``+=`` included), a container on it
        filled -- ``self.<field>.append(x)``, ``self.<field>[k] = x`` -- and,
        through a call on ``self``, what the called method writes.  So
        ``set_fatal()`` is ``status_code = self.rc_fatal`` at the call that
        makes it, which is what a reader of ``status_code`` needs.
        """
        found: list[tuple[str, ast.expr, ast.AST, Target]] = []
        for node in own_nodes(target.func):
            if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                value = node.value
                if value is None:
                    continue
                for held in targets:
                    field = _self_field(held)
                    if field:
                        found.append((field, value, node, target))
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                field = _self_field(node.func.value)
                if field and node.func.attr in FILLERS and node.args:
                    found.append((field, node.args[-1], node, target))
                elif (
                    isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "self"
                    and depth < 3
                    and target.cls is not None
                ):
                    for inner in self._method_of((target.file, target.cls), node.func.attr).targets:
                        found.extend(self.writes(inner, depth + 1))
        return found


#: The methods that put a value into a container.  The removals take one out
#: and so answer no question about where a value came from.
FILLERS = ("append", "add", "update", "extend", "setdefault", "insert")


def _parameter_names(func: ast.AST) -> set[str]:
    args = getattr(func, "args", None)
    if args is None:
        return set()
    return {a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)}


def _rebinds(func: ast.AST, name: str) -> bool:
    """Whether *func*'s own body binds *name* anywhere -- so a parameter's value may not be the one handed in."""
    return any(
        isinstance(node, ast.Name) and node.id == name and isinstance(node.ctx, (ast.Store, ast.Del))
        for node in own_nodes(func)
    )


def _self_field(node: ast.AST) -> str:
    """``status_code`` for ``self.status_code``, ``x`` for ``self.x[k]``; else ``""``."""
    if isinstance(node, ast.Subscript):
        node = node.value
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self":
        return node.attr
    return ""


def _unquoted(annotation: ast.expr) -> ast.expr:
    """"SiteMapper" as the expression it quotes; anything else as it is."""
    if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
        try:
            return ast.parse(annotation.value.strip(), mode="eval").body
        except SyntaxError:
            return annotation
    return annotation

def _element_annotation(annotation: ast.expr) -> Optional[ast.expr]:
    """The element type a container annotation states: ``dict[K, V]`` -> ``V``, ``list[T]`` -> ``T``."""
    annotation = _unquoted(annotation)
    if not isinstance(annotation, ast.Subscript):
        return None
    wrapper = annotation.value
    name = wrapper.attr if isinstance(wrapper, ast.Attribute) else getattr(wrapper, "id", "")
    inner = annotation.slice
    if name in ("dict", "Dict", "Mapping", "MutableMapping", "defaultdict", "OrderedDict"):
        return inner.elts[1] if isinstance(inner, ast.Tuple) and len(inner.elts) == 2 else None
    if name in ("list", "List", "Sequence", "set", "Set", "frozenset", "Iterable"):
        return inner
    return None


#: What a context manager is declared to yield its element through.
_YIELDING = frozenset(
    {"Iterator", "Generator", "ContextManager", "AbstractContextManager", "Iterable"}
)


class Types:
    """What class the code *declares* a receiver to be.

    Read the way a type checker reads it -- an annotation, a constructor, the
    return annotation of the method called -- and nothing else.  The walk
    needs it for one question, which system is on the other side of a call:
    ``self.site_mapper.getSite(x)`` and ``self.siteMapper.getSite(x)`` are one
    interface spelled two ways, and matching the spelling classified one and
    walked straight through the other.  What the object *is* does not depend
    on what someone called it.

    Its own parse cache, apart from the walk's: reading a base class or the
    interface's own module to learn a type is not a step on the path, so it
    does not spend the walk's file budget.  The frame's own module is taken
    from the walk, because ``pathcond.class_of`` finds a method by identity.
    """

    #: How deep a declaration is followed through fields, locals and calls.
    #: JEDI's ``self.siteMapper`` is four links from what declares it -- the
    #: field's assignment, the call, ``self.taskBufferIF``'s assignment, the
    #: ``__init__`` parameter's annotation -- and each of those is two levels
    #: here, so four stopped one short of the answer.
    DEPTH = 8

    #: How many classes a lineage holds before it stops.  ``DBProxy`` is
    #: thirteen mixins and itself; at twelve the last two were cut, and 38
    #: ``proxy.<method>()`` calls read as "the declared class has no such
    #: method".
    LINEAGE = 40

    def __init__(self, roots: dict[str, Path], walked: dict[str, tuple[str, ast.Module]]):
        self.roots = roots
        self.walked = walked
        self.parsed: dict[str, Optional[ast.Module]] = {}
        self._imports: dict[str, dict[str, tuple[str, str]]] = {}
        self._assigned: dict[int, dict[str, list[ast.expr]]] = {}
        # (file, id(func), name, depth) -> (func, class): see ``of_expression``.
        self._names: dict[tuple, tuple[ast.AST, Optional[tuple[str, ast.ClassDef]]]] = {}
        self._files: dict[int, str] = {}

    def assigned(self, func: ast.AST) -> dict[str, list[ast.expr]]:
        """``pathcond.assigned_expressions``, once per function."""
        if id(func) not in self._assigned:
            self._assigned[id(func)] = pathcond.assigned_expressions(func)
        return self._assigned[id(func)]

    # -- modules -----------------------------------------------------------

    def tree(self, file: str) -> Optional[ast.Module]:
        if file in self.walked:
            return self.walked[file][1]
        if file not in self.parsed:
            package, _, relative = file.partition("/")
            root = self.roots.get(package)
            tree = None
            if root is not None and relative:
                path = root / relative
                try:
                    stat = path.stat()
                except OSError:
                    stat = None
                key = (str(path), stat.st_mtime_ns, stat.st_size) if stat else None
                if key is not None and key in _PARSED:
                    tree = _PARSED[key]
                elif key is not None:
                    try:
                        tree = ast.parse(path.read_text(errors="replace"))
                    except (OSError, SyntaxError, ValueError):
                        tree = None
                    if tree is not None:
                        pathcond.attach_parents(tree)
                    _PARSED[key] = tree
            self.parsed[file] = tree
        return self.parsed[file]

    def file_of(self, node: ast.AST) -> Optional[str]:
        """The corpus file *node* belongs to, if it is in a tree this reader holds."""
        module = node
        while getattr(module, "parent", None) is not None:
            module = module.parent  # type: ignore[attr-defined]
        if not isinstance(module, ast.Module):
            return None
        if id(module) not in self._files:
            self._files.update({id(tree): file for file, (_sha, tree) in self.walked.items()})
            self._files.update({id(tree): file for file, tree in self.parsed.items() if tree is not None})
        return self._files.get(id(module))

    def corpus_files(self) -> list[str]:
        """Every module of the corpus, as the map spells a file; tests left out."""
        found: list[str] = []
        for package, root in sorted(self.roots.items()):
            for path in sorted(root.rglob("*.py")):
                file = f"{package}/{path.relative_to(root)}"
                if "/test" not in file:
                    found.append(file)
        return found

    def enclosing_function(self, file: str, node: ast.AST) -> Optional[ast.AST]:
        """The function *node* is nested in, or ``None`` at module or class scope."""
        parent = getattr(node, "parent", None)
        while parent is not None:
            if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return parent
            if isinstance(parent, (ast.ClassDef, ast.Module)):
                return None
            parent = getattr(parent, "parent", None)
        return None

    def method_class(self, file: str, func: ast.AST) -> Optional[tuple[str, ast.ClassDef]]:
        """The class whose ``self`` *func* reads: its own, or its enclosing method's.

        ``getJobs`` builds its ``WHERE`` in a closure, ``build_where_clause``,
        which calls ``self.construct_where_clause`` -- the ``self`` of the
        method it is nested in.
        """
        node: Optional[ast.AST] = func
        while node is not None:
            parent = getattr(node, "parent", None)
            if isinstance(parent, ast.ClassDef) and node in parent.body:
                return file, parent
            if isinstance(parent, ast.Module) or parent is None:
                break
            node = parent
        tree = self.tree(file)
        if tree is not None:
            cls = pathcond.class_of(tree, func)
            if cls is not None:
                return file, cls
        return None

    def imported_from_outside(self, file: str, name: str) -> bool:
        """Whether *name* in *file* is imported from a module this corpus does not hold."""
        held = self.imports(file).get(name)
        if held is None:
            return False
        module = held[0]
        return module.split(".")[0] not in self.roots

    def module_named(self, file: str, expression: ast.expr) -> Optional[str]:
        """The file of the module *expression* names in *file*, when it names one."""
        if not isinstance(expression, ast.Name):
            return None
        held = self.imports(file).get(expression.id)
        if held is None:
            return None
        module, attribute = held
        if not attribute:
            return self.module_file(module)
        return self.module_file(f"{module}.{attribute}")

    def module_file(self, dotted: str) -> Optional[str]:
        """``pandaserver.brokerage.SiteMapper`` -> its file, when the tree holds it."""
        parts = dotted.split(".")
        if not parts or parts[0] not in self.roots:
            return None
        for candidate in ("/".join(parts) + ".py", "/".join(parts) + "/__init__.py"):
            if self.tree(candidate) is not None:
                return candidate
        return None

    def imports(self, file: str) -> dict[str, tuple[str, str]]:
        """``{local name: (module, attribute)}`` for every import in *file*.

        Every import, wherever it sits: ``if TYPE_CHECKING:`` is where PanDA
        puts the ones only an annotation needs, and those are the ones this
        reads.  A relative import is resolved against the file's own package.
        """
        if file in self._imports:
            return self._imports[file]
        found: dict[str, tuple[str, str]] = {}
        tree = self.tree(file)
        package = file[: -len(".py")].split("/")[:-1] if file.endswith(".py") else []
        for node in ast.walk(tree) if tree is not None else ():
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.asname:
                        found[alias.asname] = (alias.name, "")
                    else:
                        head = alias.name.split(".")[0]
                        found.setdefault(head, (head, ""))
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base = package[: len(package) - (node.level - 1)] if node.level > 1 else package
                    module = ".".join([*base, *([node.module] if node.module else [])])
                else:
                    module = node.module or ""
                for alias in node.names:
                    found[alias.asname or alias.name] = (module, alias.name)
        self._imports[file] = found
        return found

    # -- classes -----------------------------------------------------------

    def class_named(self, file: str, expression: ast.expr, depth: int = 0) -> Optional[tuple[str, ast.ClassDef]]:
        """The class *expression* names in *file*: declared there, or imported."""
        if depth > self.DEPTH:
            return None
        tree = self.tree(file)
        if tree is None:
            return None
        if isinstance(expression, ast.Name):
            for node in tree.body:
                if isinstance(node, ast.ClassDef) and node.name == expression.id:
                    return file, node
            held = self.imports(file).get(expression.id)
            if held is None:
                return None
            module, attribute = held
            if not attribute:
                return None
            # ``from pkg.mod import Cls`` -- or ``from pkg import mod``, a module.
            target = self.module_file(module)
            if target is not None:
                return self.class_named(target, ast.Name(id=attribute, ctx=ast.Load()), depth + 1)
            return None
        if isinstance(expression, ast.Attribute) and isinstance(expression.value, ast.Name):
            # ``SiteMapper.SiteMapper`` -- a module imported, then its class.
            held = self.imports(file).get(expression.value.id)
            if held is None:
                return None
            module, attribute = held
            dotted = f"{module}.{attribute}" if attribute else module
            target = self.module_file(dotted)
            if target is None:
                return None
            return self.class_named(target, ast.Name(id=expression.attr, ctx=ast.Load()), depth + 1)
        return None

    def annotated(self, file: str, annotation: Optional[ast.expr]) -> Optional[tuple[str, ast.ClassDef]]:
        """The class an annotation states, through quotes and ``Optional``/``| None``."""
        if annotation is None:
            return None
        annotation = _unquoted(annotation)
        if isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.BitOr):
            sides = [
                side for side in (annotation.left, annotation.right)
                if not (isinstance(side, ast.Constant) and side.value is None)
            ]
            found = {self.annotated(file, side) for side in sides} - {None}
            return found.pop() if len(found) == 1 else None
        if isinstance(annotation, ast.Subscript):
            wrapper = annotation.value
            name = wrapper.attr if isinstance(wrapper, ast.Attribute) else getattr(wrapper, "id", "")
            if name == "Optional":
                return self.annotated(file, annotation.slice)
            if name == "Union" and isinstance(annotation.slice, ast.Tuple):
                sides = [
                    side for side in annotation.slice.elts
                    if not (isinstance(side, ast.Constant) and side.value is None)
                ]
                found = {self.annotated(file, side) for side in sides} - {None}
                return found.pop() if len(found) == 1 else None
            return None
        return self.class_named(file, annotation)

    def lineage(self, file: str, cls: ast.ClassDef) -> list[tuple[str, ast.ClassDef]]:
        """*cls* and its bases, nearest first, followed across files through imports."""
        order: list[tuple[str, ast.ClassDef]] = [(file, cls)]
        queue = [(file, base) for base in cls.bases]
        seen = {(file, cls.name)}
        while queue and len(order) < self.LINEAGE:
            where, base = queue.pop(0)
            found = self.class_named(where, base)
            if found is None or (found[0], found[1].name) in seen:
                continue
            seen.add((found[0], found[1].name))
            order.append(found)
            queue.extend((found[0], one) for one in found[1].bases)
        return order

    def method(self, file: str, cls: ast.ClassDef, name: str):
        for where, holder in self.lineage(file, cls):
            for node in holder.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
                    return where, node
        return None

    # -- declarations ------------------------------------------------------

    def of_field(self, file: str, cls: ast.ClassDef, field: str, depth: int) -> Optional[tuple[str, ast.ClassDef]]:
        """``self.<field>``'s class: its annotation in the class or a base, else what it is assigned."""
        lineage = self.lineage(file, cls)
        for where, holder in lineage:
            for node in holder.body:
                if (
                    isinstance(node, ast.AnnAssign)
                    and isinstance(node.target, ast.Name)
                    and node.target.id == field
                ):
                    return self.annotated(where, node.annotation)
            for method in pathcond.methods_of(holder):
                for node in ast.walk(method):
                    if (
                        isinstance(node, ast.AnnAssign)
                        and isinstance(node.target, ast.Attribute)
                        and isinstance(node.target.value, ast.Name)
                        and node.target.value.id == "self"
                        and node.target.attr == field
                    ):
                        return self.annotated(where, node.annotation)
        if depth >= self.DEPTH:
            return None
        found: set[tuple[str, str]] = set()
        held: dict[tuple[str, str], tuple[str, ast.ClassDef]] = {}
        for where, holder in lineage:
            for method in pathcond.methods_of(holder):
                for expression in pathcond.attribute_expressions(method, field):
                    if isinstance(expression, ast.Constant) and expression.value is None:
                        continue
                    one = self.of_expression(where, method, expression, depth + 1)
                    if one is None:
                        return None
                    found.add((one[0], one[1].name))
                    held[(one[0], one[1].name)] = one
        return held[found.pop()] if len(found) == 1 else None

    def _with_held(self, file: str, func: ast.AST, name: str, depth: int) -> Optional[tuple[str, ast.ClassDef]]:
        """The class ``with <call> as <name>`` binds, from what the call is declared to yield.

        Every ``TaskBuffer`` method reaches the database through ``with
        self.proxyPool.get() as proxy``, and ``get`` is declared ``->
        Iterator["DBProxy.DBProxy"]``: the object is the element, not the
        iterator.
        """
        for node in ast.walk(func):
            if not isinstance(node, (ast.With, ast.AsyncWith)):
                continue
            for item in node.items:
                held = item.optional_vars
                if not (isinstance(held, ast.Name) and held.id == name):
                    continue
                context = item.context_expr
                if not (isinstance(context, ast.Call) and isinstance(context.func, ast.Attribute)):
                    return self.of_expression(file, func, context, depth + 1)
                receiver = context.func.value
                if isinstance(receiver, ast.Name) and receiver.id == "self":
                    owner = self.method_class(file, func)
                else:
                    owner = self.of_expression(file, func, receiver, depth + 1)
                if owner is None:
                    return None
                found = self.method(owner[0], owner[1], context.func.attr)
                if found is None:
                    return None
                where, method = found
                returned = _unquoted(method.returns) if method.returns is not None else None
                if isinstance(returned, ast.Subscript):
                    wrapper = returned.value
                    wrapped = wrapper.attr if isinstance(wrapper, ast.Attribute) else getattr(wrapper, "id", "")
                    if wrapped in _YIELDING:
                        inner = returned.slice
                        if isinstance(inner, ast.Tuple) and inner.elts:
                            inner = inner.elts[0]
                        return self.annotated(where, inner)
                return self.annotated(where, returned)
        return None

    def _annotation_of(
        self, file: str, func: ast.AST, expression: ast.expr, seen: frozenset = frozenset()
    ) -> Optional[tuple[str, ast.expr]]:
        """The annotation *expression* was declared with: a field's, a parameter's or a local's."""
        if (
            isinstance(expression, ast.Attribute)
            and isinstance(expression.value, ast.Name)
            and expression.value.id == "self"
        ):
            holder = self.method_class(file, func)
            if holder is None:
                return None
            for where, cls in self.lineage(*holder):
                for node in cls.body:
                    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == expression.attr:
                        return where, node.annotation
                for method in pathcond.methods_of(cls):
                    for node in ast.walk(method):
                        if (
                            isinstance(node, ast.AnnAssign)
                            and isinstance(node.target, ast.Attribute)
                            and isinstance(node.target.value, ast.Name)
                            and node.target.value.id == "self"
                            and node.target.attr == expression.attr
                        ):
                            return where, node.annotation
            return None
        if isinstance(expression, ast.Name):
            args = getattr(func, "args", None)
            for argument in (*args.posonlyargs, *args.args, *args.kwonlyargs) if args else ():
                if argument.arg == expression.id and argument.annotation is not None:
                    return file, argument.annotation
            for node in ast.walk(func):
                if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == expression.id:
                    return file, node.annotation
            # ``target = self.nuclei`` or ``target = self.satellites``: an
            # alias carries what every value it is bound to was declared as.
            bound = self.assigned(func).get(expression.id, [])
            if bound and expression.id not in seen:
                found = [
                    self._annotation_of(file, func, value, seen | {expression.id}) for value in bound
                ]
                texts = {ast.unparse(one[1]) for one in found if one is not None}
                if len(texts) == 1 and all(one is not None for one in found):
                    return found[0]
        return None

    def _function_named(self, file: str, name: str) -> Optional[tuple[str, ast.AST]]:
        """The module-level function *name* is in *file*: defined there or imported."""
        tree = self.tree(file)
        if tree is None:
            return None
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
                return file, node
        held = self.imports(file).get(name)
        if held is None or not held[1]:
            return None
        target = self.module_file(held[0])
        if target is None or target == file:
            return None
        other = self.tree(target)
        for node in other.body if other is not None else ():
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == held[1]:
                return target, node
        return None

    def _outside_instance(self, file: str, name: str, depth: int) -> Optional[tuple[str, ast.ClassDef]]:
        """The class of a name this function does not bind: a module's own instance.

        ``taskBuffer = TaskBuffer()`` at the foot of ``TaskBuffer.py`` and
        ``rucioAPI = RucioAPI()`` at the foot of ``ddm.py`` are what every
        daemon imports and calls; the module's own assignment says what it is.
        """
        if depth > self.DEPTH:
            return None
        tree = self.tree(file)
        if tree is None:
            return None
        for node in tree.body:
            # ``global_task_buffer: TaskBuffer = None``: the annotation is the
            # declaration, whatever the placeholder value is.
            if (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.target.id == name
            ):
                declared = self.annotated(file, node.annotation)
                if declared is not None:
                    return declared
        values = [
            node.value
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
        ]
        if values:
            found = {self.of_expression(file, tree, value, depth + 1) for value in values}
            return found.pop() if len(found) == 1 else None
        held = self.imports(file).get(name)
        if held is None or not held[1]:
            return None
        target = self.module_file(held[0])
        if target is None or target == file:
            return None
        return self._outside_instance(target, held[1], depth + 1)

    def _of_name(self, file: str, func: ast.AST, name: str, depth: int) -> Optional[tuple[str, ast.ClassDef]]:
        """:meth:`of_expression` for a bare name, read in *func*."""
        args = getattr(func, "args", None)
        for argument in (*args.posonlyargs, *args.args, *args.kwonlyargs) if args else ():
            if argument.arg == name:
                return self.annotated(file, argument.annotation)
        for node in ast.walk(func):
            if (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.target.id == name
            ):
                return self.annotated(file, node.annotation)
        # Every assignment, less the ``None`` a name is cleared with:
        # ``siteMapper = SiteMapper(taskBuffer)`` and, at the end of the
        # daemon, ``siteMapper = None``.  The evidence for a local is all in
        # this function, so it is read rather than asked for; the answer
        # is a class only where every assignment says the same one.
        bound = [
            value
            for value in self.assigned(func).get(name, [])
            if not (isinstance(value, ast.Constant) and value.value is None)
        ]
        held = self._with_held(file, func, name, depth)
        if not bound and held is None and not isinstance(func, ast.Module):
            return self._outside_instance(file, name, depth)
        found = {self.of_expression(file, func, value, depth + 1) for value in bound}
        if held is not None:
            found.add(held)
        return found.pop() if len(found) == 1 else None

    def of_expression(self, file: str, func: ast.AST, expression: ast.expr, depth: int = 0) -> Optional[tuple[str, ast.ClassDef]]:
        """The class *expression*, read in *func* of *file*, is declared to be."""
        if depth > self.DEPTH:
            return None
        tree = self.tree(file)
        if tree is None:
            return None
        if isinstance(expression, ast.Name):
            # What a name is depends on the function, the name and how deep
            # the question already is -- not on which ``Name`` asked.
            # ``JobGenerator`` rebinds locals from calls on locals that are
            # rebound too, and reading each afresh cost the product of their
            # assignment counts: a census stopped in that one file for hours.
            # The function is kept with the answer, so its id is not reused.
            key = (file, id(func), expression.id, depth)
            hit = self._names.get(key)
            if hit is not None and hit[0] is func:
                return hit[1]
            answer = self._of_name(file, func, expression.id, depth)
            self._names[key] = (func, answer)
            return answer
        if (
            isinstance(expression, ast.Attribute)
            and isinstance(expression.value, ast.Name)
            and expression.value.id == "self"
        ):
            cls = pathcond.class_of(tree, func)
            return self.of_field(file, cls, expression.attr, depth + 1) if cls is not None else None
        if isinstance(expression, ast.Subscript):
            # ``self.nuclei[name]``, where ``self.nuclei: dict[str, NucleusSpec]``.
            container = self._annotation_of(file, func, expression.value)
            if container is not None:
                where, annotation = container
                element = _element_annotation(annotation)
                if element is not None:
                    return self.annotated(where, element)
            return None
        if isinstance(expression, ast.Call):
            called = expression.func
            constructed = self.class_named(file, called) if isinstance(called, (ast.Name, ast.Attribute)) else None
            if constructed is not None:
                return constructed
            if isinstance(called, ast.Name):
                # ``get_task_utils_module(self)``, declared ``-> TaskUtilsModule``.
                function = self._function_named(file, called.id)
                if function is not None:
                    where, definition = function
                    return self.annotated(where, definition.returns)
                return None
            if isinstance(called, ast.Attribute):
                receiver = called.value
                if isinstance(receiver, ast.Name) and receiver.id == "self":
                    owner = self.method_class(file, func)
                else:
                    owner = self.of_expression(file, func, receiver, depth + 1)
                if owner is None:
                    return None
                found = self.method(owner[0], owner[1], called.attr)
                if found is None:
                    return None
                where, method = found
                return self.annotated(where, method.returns)
        return None

    # -- what a value can be, where nothing declares it --------------------

    def candidates(
        self, file: str, func: ast.AST, expression: ast.expr, depth: int = 0
    ) -> tuple[frozenset[tuple[str, str]], bool]:
        """``({(file, class name)}, open)``: the classes *expression* can be an instance of.

        Asked where the declaration says nothing -- no annotation, or ``Any``.
        ``AdderGen`` declares ``self.adder_plugin: Any`` and builds it as
        ``self.adder_plugin_class(...)``, where ``adder_plugin_class`` is what
        ``get_plugin_class`` returns: ``AdderAtlasPlugin``, or a class the
        configuration names.  Read from the values rather than the
        declaration, that is ``{AdderAtlasPlugin}`` and *open* -- there is
        another way, and this reading cannot say which class it is.
        """
        declared = self.of_expression(file, func, expression, 0)
        if declared is not None:
            return frozenset({(declared[0], declared[1].name)}), False
        return self._values(file, func, expression, depth, instances=True, seen=frozenset())

    def class_def(self, file: str, name: str) -> Optional[ast.ClassDef]:
        tree = self.tree(file)
        for node in ast.walk(tree) if tree is not None else ():
            if isinstance(node, ast.ClassDef) and node.name == name:
                return node
        return None

    def _values(
        self,
        file: str,
        func: ast.AST,
        expression: ast.expr,
        depth: int,
        *,
        instances: bool,
        seen: frozenset,
    ) -> tuple[frozenset[tuple[str, str]], bool]:
        """Classes an expression holds -- instances of, or the class objects themselves."""
        if depth > self.DEPTH:
            return frozenset(), True
        key = (file, id(func), ast.unparse(expression), instances)
        if key in seen:
            return frozenset(), False
        seen = seen | {key}
        found: set[tuple[str, str]] = set()
        opened = False

        def add(result: tuple[frozenset, bool]) -> None:
            nonlocal opened
            found.update(result[0])
            opened = opened or result[1]

        if isinstance(expression, ast.Constant) and expression.value is None:
            return frozenset(), False
        if isinstance(expression, ast.Call):
            if instances:
                # An instance of whatever class the callee is: ``Cls(...)``,
                # or ``self.plugin_class(...)`` holding a class.
                add(self._values(file, func, expression.func, depth + 1, instances=False, seen=seen))
                return frozenset(found), opened
            # A call that returns a class: what its body returns.
            owner = self._owner_of(file, func, expression.func)
            if owner is None:
                return frozenset(), True
            where, method = owner
            for node in ast.walk(method):
                if isinstance(node, ast.Return) and node.value is not None:
                    add(self._values(where, method, node.value, depth + 1, instances=False, seen=seen))
            return frozenset(found), opened
        if isinstance(expression, ast.Name):
            if not instances:
                named = self.class_named(file, expression)
                if named is not None:
                    return frozenset({(named[0], named[1].name)}), False
            bound = [
                value
                for value in self.assigned(func).get(expression.id, [])
                if not (isinstance(value, ast.Constant) and value.value is None)
            ]
            if not bound:
                return frozenset(), True
            for value in bound:
                add(self._values(file, func, value, depth + 1, instances=instances, seen=seen))
            return frozenset(found), opened
        if (
            isinstance(expression, ast.Attribute)
            and isinstance(expression.value, ast.Name)
            and expression.value.id == "self"
        ):
            holder = self.method_class(file, func)
            if holder is None:
                return frozenset(), True
            any_value = False
            for where, cls in self.lineage(*holder):
                for method in pathcond.methods_of(cls):
                    for value in pathcond.attribute_expressions(method, expression.attr):
                        if isinstance(value, ast.Constant) and value.value is None:
                            continue
                        any_value = True
                        add(self._values(where, method, value, depth + 1, instances=instances, seen=seen))
            return frozenset(found), opened or not any_value
        if isinstance(expression, ast.Attribute):
            # ``obj.field`` where ``obj`` is one of some classes: what each assigns it.
            owners, owner_open = self.candidates(file, func, expression.value, depth + 1)
            opened = owner_open
            for where, name in owners:
                cls = self.class_def(where, name)
                if cls is None:
                    continue
                typed = self.of_field(where, cls, expression.attr, depth + 1)
                if typed is not None and instances:
                    found.add((typed[0], typed[1].name))
                    continue
                for place, holder in self.lineage(where, cls):
                    for method in pathcond.methods_of(holder):
                        for value in pathcond.attribute_expressions(method, expression.attr):
                            if isinstance(value, ast.Constant) and value.value is None:
                                continue
                            add(self._values(place, method, value, depth + 1, instances=instances, seen=seen))
            return frozenset(found), opened
        return frozenset(), True

    def _owner_of(self, file: str, func: ast.AST, called: ast.expr) -> Optional[tuple[str, ast.AST]]:
        """The function *called* names, for a call on ``self`` or a plain name."""
        if (
            isinstance(called, ast.Attribute)
            and isinstance(called.value, ast.Name)
            and called.value.id == "self"
        ):
            holder = self.method_class(file, func)
            if holder is None:
                return None
            return self.method(holder[0], holder[1], called.attr)
        return None

