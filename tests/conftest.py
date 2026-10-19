"""Make the project root importable for the test suite.

Running ``python -m pytest`` from the project root already puts the root on
``sys.path``. This file also allows running ``pytest`` directly from inside
the ``tests/`` folder.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
