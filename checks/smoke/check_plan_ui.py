"""Boundary input regression; in-memory configuration only."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from checks.test_boundary_editor import BoundaryTest

if __name__=='__main__':
    import unittest
    unittest.main(module='checks.test_boundary_editor')
