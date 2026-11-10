"""Shared recognizers for code that writes state through the Django ORM.

PanDA's write recognizers read SQL text and infer a row's class from how it is
used.  A Django project mostly writes through the ORM, and there the code
*names* the class: ``ProdTask.objects.filter(...).update(status='done')``, or
``task.status = new`` on a ``task`` bound from ``ProdTask.objects.get()``.  The
models declare their table, fields and allowed values, so subjects and their
vocabulary come from the declarations rather than from usage.

The package is plugin-independent and additive: a plugin for a Django code
base calls

* :func:`~bamboo.codemap.orm.schema.read_models` for the models,
* :func:`~bamboo.codemap.orm.writes.scan_function` for each function's writes
  and filter reads,
* :func:`~bamboo.codemap.orm.resolve.resolve` for each written value, which
  delegates to the shared value machinery (``pathcond.reaching_values``,
  ``values.resolver``) and adds narrowing by a membership guard, and
* :func:`~bamboo.codemap.orm.assemble.assemble` for the subjects, entities and
  junctions of its :class:`~bamboo.codemap.models.MapFragment`.

What stays in the plugin is what is project-specific: which packages are read,
raw-SQL forms, boundaries to other systems.
"""
