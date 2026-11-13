"""Shared Django-ORM recognizers (bamboo.codemap.orm), on a small project of their own."""

from __future__ import annotations

import ast
import textwrap

import pytest

from bamboo.codemap import gates
from bamboo.codemap.models import JunctionNode, MapFragment
from bamboo.codemap.orm.assemble import FieldWrite, assemble, write_condition
from bamboo.codemap.orm.resolve import resolve, settler
from bamboo.codemap.orm.schema import module_constants, read_models
from bamboo.codemap.orm.writes import scan_function
from bamboo.codemap.panda import pathcond

MODELS = """
from django.db import models

ORDER_STATES = [("open", "Open"), ("paid", "Paid"), ("shipped", "Shipped")]


class Stamped(models.Model):
    state = models.CharField(max_length=10, choices=[("new", "New"), ("old", "Old")], default="new")

    class Meta:
        abstract = True


class Order(Stamped):
    status = models.CharField(max_length=10, choices=ORDER_STATES, default="open")
    note = models.TextField()

    class Meta:
        db_table = "shop_orders"

    def pay(self):
        self.status = "paid"


class Ticket(models.Model):
    class Priority(models.TextChoices):
        LOW = "low", "Low"
        HIGH = "high", "High"

    KIND_CHOICES = [("bug", "Bug"), ("task", "Task")]

    priority = models.CharField(max_length=8, choices=Priority.choices)
    kind = models.CharField(max_length=8, choices=KIND_CHOICES)
    title = models.CharField(max_length=80)


class Invoice(models.Model):
    status = models.CharField(max_length=10, default="draft")


class Plain:
    status = "not a model"
"""

VIEWS = """
from shop.models import Order, Ticket

ALLOWED = ("paid", "shipped")


def ship(pk):
    order = Order.objects.get(pk=pk)
    if order.status == "paid":
        order.status = "shipped"
    order.save()


def from_request(order, data):
    state = (data.get("status") or "").strip()
    if state not in ALLOWED:
        raise ValueError(state)
    order.status = state


def from_rows(rows):
    kinds = {k for k, _ in Ticket.KIND_CHOICES}
    for row in rows:
        kind = str(row.get("kind"))
        if not row or kind not in kinds:
            continue
        Ticket.objects.update_or_create(title=row["title"], defaults={"kind": kind})


def rebound(order, data):
    state = data["status"]
    if state not in ALLOWED:
        return
    state = data["other"]
    order.status = state


def chosen(pk, ok):
    order = Order.objects.get(pk=pk)
    order.status = "paid" if ok else "open"
    if ok:
        new = "shipped"
    else:
        new = "open"
    order.status = new


def enum_and_copy(pk):
    Ticket.objects.filter(pk=pk).update(priority=Ticket.Priority.HIGH)
    Order.objects.filter(pk=pk).update(status=F("state"))


def bulk(orders):
    Order.objects.bulk_update(orders, ["status", "note"])


def make():
    Order.objects.create(note="x")
    Order(note="y").save()
    for o in Order.objects.filter(status__in=["open"]):
        o.state = "old"


def structural(o):
    o.note = "z"
    o.status = "open"


def heuristic(orders):
    orders.status = "paid"


def unknown(x):
    x.status = "open"


class Helper:
    def __init__(self):
        self.status = "ignored"
"""


def _parse(text: str, rel: str, module: str):
    tree = ast.parse(textwrap.dedent(text), filename=rel)
    pathcond.attach_parents(tree)
    return rel, module, tree


@pytest.fixture(scope="module")
def project():
    models_src = _parse(MODELS, "shop/models.py", "shop.models")
    views_src = _parse(VIEWS, "shop/views.py", "shop.views")
    models = read_models([models_src, views_src])
    return models, views_src[2]


def _scan(project, name):
    models, tree = project
    func = next(
        f for f in tree.body if isinstance(f, ast.FunctionDef) and f.name == name
    )
    constants = module_constants(tree)
    scan = scan_function(func, None, models, constants)
    settle = settler(models, constants)
    resolved = [
        (
            w,
            resolve(w.value, w.at, func, settle, models, model=w.model),
            write_condition(w.at),
        )
        for w in scan.writes
    ]
    return scan, resolved


def _outcomes(resolved, field="status"):
    return sorted(
        (r.outcome, r.tier)
        for w, arms, _c in resolved
        if w.field == field
        for r in arms
    )


# --------------------------------------------------------------------------- schema


def test_models_fields_choices_and_tables(project):
    models, _ = project
    assert set(models) == {"Stamped", "Order", "Ticket", "Invoice"}  # Plain is no model
    order = models["Order"]
    assert order.db_table == "shop_orders"
    assert order.fields["status"] == ["open", "paid", "shipped"]
    assert order.fields["state"] == ["new", "old"]  # inherited from the abstract base
    assert order.defaults == {"state": "new", "status": "open"}
    assert order.state_fields() == {"status", "state"}
    assert models["Stamped"].abstract


def test_enumerations_and_class_choice_lists(project):
    models, _ = project
    ticket = models["Ticket"]
    assert ticket.fields["priority"] == ["low", "high"]  # TextChoices.choices
    assert ticket.fields["kind"] == ["bug", "task"]  # class-level list
    assert ticket.db_table == "shop_ticket"  # Django's default
    assert ticket.state_fields() == {
        "priority",
        "kind",
    }  # choices= makes a field state-like


# --------------------------------------------------------------------------- write forms and attribution


def test_bound_object_is_certain_with_conditions(project):
    _scan_, resolved = _scan(project, "ship")
    [(w, arms, conds)] = resolved
    assert (w.model, w.attribution, w.form) == ("Order", "certain", "attribute")
    assert [(a.outcome, a.tier) for a in arms] == [("shipped", 1)]
    assert conds == ["order.status == 'paid'"]


def test_create_constructor_and_loop_binding(project):
    scan, resolved = _scan(project, "make")
    forms = sorted((w.model, w.field, w.form) for w in scan.writes)
    assert forms == [
        ("Order", "state", "attribute"),
        ("Order", "state", "orm.<init>(default)"),
        ("Order", "state", "orm.create(default)"),
        ("Order", "status", "orm.<init>(default)"),
        ("Order", "status", "orm.create(default)"),
    ]
    assert _outcomes(resolved) == [("open", 1), ("open", 1)]
    assert scan.selected[("Order", "status")] == {"open"}


def test_bulk_update_names_fields_without_a_value(project):
    _scan_, resolved = _scan(project, "bulk")
    [(w, arms, _c)] = resolved  # "note" is not state-like
    assert (w.field, w.form) == ("status", "orm.bulk_update")
    assert arms[0].tier == 2


def test_attribution_ladder(project):
    for name, expected in [
        ("structural", "structural"),
        ("heuristic", "heuristic"),
        ("unknown", "unresolved"),
    ]:
        scan, _ = _scan(project, name)
        [w] = [w for w in scan.writes if w.field == "status"]
        assert w.attribution == expected, name
    scan, _ = _scan(project, "structural")
    assert scan.writes[-1].structural == "Order"
    assert _scan(project, "unknown")[0].writes[0].model is None


def test_self_inside_model_and_non_model_class(project):
    models, _ = project
    tree = _parse(MODELS, "shop/models.py", "shop.models")[2]
    order_cls = next(
        c for c in tree.body if isinstance(c, ast.ClassDef) and c.name == "Order"
    )
    pay = next(f for f in order_cls.body if isinstance(f, ast.FunctionDef))
    scan = scan_function(pay, "Order", models, module_constants(tree))
    assert [(w.model, w.attribution) for w in scan.writes] == [("Order", "certain")]
    views = project[1]
    helper = next(
        c for c in views.body if isinstance(c, ast.ClassDef) and c.name == "Helper"
    )
    assert scan_function(helper.body[0], "Helper", models, {}).writes == []


# --------------------------------------------------------------------------- values


def test_membership_guard_narrows_a_request_value(project):
    _scan_, resolved = _scan(project, "from_request")
    [(w, arms, conds)] = resolved
    assert [(a.outcome, a.tier, a.conditions) for a in arms] == [
        ("paid", 1, ["state == 'paid'"]),
        ("shipped", 1, ["state == 'shipped'"]),
    ]
    assert conds == ["not (state not in ALLOWED)"]


def test_disjunctive_exit_and_local_container(project):
    _scan_, resolved = _scan(project, "from_rows")
    assert _outcomes(resolved, "kind") == [("bug", 1), ("task", 1)]


def test_rebinding_after_the_guard_defeats_narrowing(project):
    _scan_, resolved = _scan(project, "rebound")
    assert _outcomes(resolved) == [("runtime(state)", 2)]


def test_conditional_expression_and_reaching_bindings(project):
    _scan_, resolved = _scan(project, "chosen")
    by_line = [[(a.outcome, a.conditions) for a in arms] for _w, arms, _c in resolved]
    assert by_line[0] == [("paid", ["ok"]), ("open", ["not (ok)"])]
    assert sorted(by_line[1]) == [("open", ["not (ok)"]), ("shipped", ["ok"])]


def test_enum_member_and_column_copy(project):
    _scan_, resolved = _scan(project, "enum_and_copy")
    assert _outcomes(resolved, "priority") == [("high", 1)]
    assert _outcomes(resolved, "status") == [("passthrough(Order.state)", 2)]


# --------------------------------------------------------------------------- assembly


def test_assembly_names_and_gates(project):
    models, tree = project
    writes: list[FieldWrite] = []
    for func in (f for f in tree.body if isinstance(f, ast.FunctionDef)):
        constants = module_constants(tree)
        settle = settler(models, constants)
        for w in scan_function(func, None, models, constants).writes:
            writes.append(
                FieldWrite(
                    model=w.model,
                    field=w.field,
                    owner=f"shop/views.py::{func.name}",
                    rel_path="shop/views.py",
                    line=w.line,
                    form=w.form,
                    attribution=w.attribution,
                    arms=resolve(w.value, w.at, func, settle, models, model=w.model),
                    path_condition=write_condition(w.at),
                    structural=w.structural,
                )
            )
    subjects, entities, junctions, unresolved = assemble(
        writes, models, "shop", "shop test", {}
    )
    assert unresolved == 1  # unknown()
    ship = next(j for j in junctions if j.owner.endswith("::ship"))
    assert ship.name == JunctionNode.make_name(
        "shop", "Order.status", "shop/views.py::ship"
    )
    order = next(e for e in entities if e.spec_class == "Order")
    assert "shop/views.py::make" in order.created_by
    assert "shop/views.py::ship" in order.updated_by
    fragment = MapFragment(
        map_id="shop",
        derived_from="shop test",
        subjects=subjects,
        entities=entities,
        junctions=junctions,
    )
    results = {r.gate: r for r in gates.run_all(fragment)}
    for name in (
        "map-references-resolve",
        "branch-positions-are-known",
        "map-identities-are-distinct",
    ):
        assert results[name].passed, results[name].failures
    status = next(s for s in subjects if s.name == "Order.status")
    assert status.vocabulary == ["open", "paid", "shipped"]
