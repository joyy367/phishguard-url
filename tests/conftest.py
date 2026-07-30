"""
conftest.py
-----------
Shared pytest fixtures and path configuration.
Adds the project root to sys.path so all app.* imports resolve correctly
regardless of where pytest is invoked from.
"""

import os
import sys

# Add the project root to sys.path
ROOT = os.path.join(os.path.dirname(__file__), "..")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
