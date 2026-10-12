"""Field tests: the map and the querier run on real PanDA tasks and jobs, scored by how far they get alone.

A case is one question about one task or job -- why its state is what it is,
why it moved on late, or why it does not move -- held as data: the question,
the rows and logs it is answered from, and the PanDA version (the pin) whose
code wrote those logs.  :mod:`bamboo.fieldtest.run` runs the querier on cases
with no hand file and records where each run ended: the steps before the first
``STOP`` ran with no one choosing.  :mod:`bamboo.fieldtest.census` sizes, from a
map dump, each class of input the querier asks PanDA for, so that the API that
feeds the cases is shaped by classes and not by the cases at hand.

The cases, their inputs and the runs are machine-local and stay out of the
repository, in a directory named by ``--home`` or ``FIELDTEST_HOME``:

``fieldtest.json``
    ``python`` (the interpreter the querier runs under; default: this one) and
    ``cwd`` (where the querier runs; default: the home).
``cases/**/*.json``
    One case each (:func:`bamboo.fieldtest.run.load_case`).
``pairs/*.json``, ``pairs/current.json``
    The pairs a case can run on, and the one each pin runs on now.
``runs/``
    What the runs wrote.
"""
