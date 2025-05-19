#!/usr/bin/env python3
"""
List all modules and functions in evlib

Note: The module structure has been updated:
- 'events' module has been renamed to 'core'
- All internal modules now use 'ev_' prefix
"""
import evlib

# List all attributes in the evlib module
print("All attributes:", dir(evlib))

# Filter out modules (non-callable attributes that don't start with '_')
modules = [
    m for m in dir(evlib) if not m.startswith("_") and not callable(getattr(evlib, m)) and m != "evlib"
]
print("\nModules:", modules)

# List functions at the root level
functions = [f for f in dir(evlib) if not f.startswith("_") and callable(getattr(evlib, f))]
print("\nRoot functions:", functions)

# List functions in each module
for module_name in modules:
    module = getattr(evlib, module_name)
    module_functions = [f for f in dir(module) if not f.startswith("_") and callable(getattr(module, f))]
    print(f"\n{module_name} functions:", module_functions)
