import evlib

print("\nImporting directly from evlib:")
try:
    # Try directly accessing evlib.formats.load_events_py
    print("evlib.formats:", dir(evlib.formats))
    print("evlib.formats.load_events_py exists:", hasattr(evlib.formats, "load_events_py"))
    print("Type of evlib.formats.load_events_py:", type(evlib.formats.load_events_py))
    
    # Try importing from evlib
    from evlib.formats import load_events_py
    print("Successfully imported load_events_py from evlib.formats")
except Exception as e:
    print(f"Error: {e}")
    
print("\nWorkaround for testing:")
try:
    # Direct access seems to work
    print("Testing direct access:")
    func = evlib.formats.load_events_py
    print("Function name:", func.__name__)
    print("Function module:", func.__module__)
except Exception as e:
    print(f"Error: {e}")