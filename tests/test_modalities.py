import importlib


def test_audio_scaffold_imports_without_optional_dependencies():
    importlib.import_module("equimo.audio")
    importlib.import_module("equimo.audio.models")
    importlib.import_module("equimo.audio.layers")
    importlib.import_module("equimo.audio.io")


def test_tabular_scaffold_imports_without_optional_dependencies():
    importlib.import_module("equimo.tabular")
    importlib.import_module("equimo.tabular.models")
    importlib.import_module("equimo.tabular.layers")


def test_old_top_level_packages_are_not_importable():
    for module_name in (
        "equimo.models",
        "equimo.layers",
        "equimo.implicit",
        "equimo.experimental",
    ):
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
        raise AssertionError(f"{module_name} should not be importable")
