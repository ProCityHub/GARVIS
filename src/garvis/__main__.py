"""Allow ``python -m garvis`` to launch the GARVIS CLI."""

from .cli import main

raise SystemExit(main())
