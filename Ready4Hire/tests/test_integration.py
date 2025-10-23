#!/usr/bin/env python3
"""
Tests de Integración - Ready4Hire v2.2.0

Tests consolidados que validan todos los componentes.
Este archivo reemplaza múltiples tests dispersos.
"""
import sys
import os

# Agregar path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from test_integration_full import *

if __name__ == "__main__":
    exit(main())
