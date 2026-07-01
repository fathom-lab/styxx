# -*- coding: utf-8 -*-
"""Package-data anchor for styxx bundled receipts (leaderboard, competence-cliff, confound boundaries).

Intentionally a *regular* package (this file exists) rather than a namespace package: on Python 3.9,
``importlib.resources.files("styxx._data")`` routes through ``importlib._common.from_package``, which
dereferences ``package.__file__``. A namespace package has ``__file__ is None`` → ``TypeError``. Shipping
this ``__init__.py`` gives the package a real ``__file__`` so ``files()`` resolves on 3.9 as it does on 3.10+.
"""
