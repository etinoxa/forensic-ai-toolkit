'''Fait plugins auto-import.'''
def _autoimport_plugins():
    import importlib
    for mod in (
        "fait.vision.embeddings.arcface",
        "fait.vision.embeddings.clip",
    ):
        try:
            importlib.import_module(mod)
        except Exception as e:
            # You can log or ignore. Avoid raising so the package still imports.
            pass

_autoimport_plugins()
