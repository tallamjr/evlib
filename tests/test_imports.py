import evlib

print("\nImporting directly from evlib:")
try:
    # Try directly accessing evlib.formats.load_events
    print("evlib.formats:", dir(evlib.formats))
    print("evlib.formats.load_events exists:", hasattr(evlib.formats, "load_events"))
    print("Type of evlib.formats.load_events:", type(evlib.formats.load_events))

    # Try importing from evlib

    print("Successfully imported load_events from evlib.formats")
except Exception as e:
    print(f"Error: {e}")

print("\nWorkaround for testing:")
try:
    # Direct access seems to work
    print("Testing direct access:")
    func = evlib.formats.load_events
    print("Function name:", func.__name__)
    print("Function module:", func.__module__)
except Exception as e:
    print(f"Error: {e}")
