# scripts/use_config.py
"""
Switch between configuration presets.

Usage:
    python scripts/use_config.py high_accuracy
    python scripts/use_config.py fast
    python scripts/use_config.py balanced
    python scripts/use_config.py path/to/custom.yaml
"""

import sys
import shutil
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/use_config.py <preset_name|path>")
        print("\nAvailable presets:")
        repo_root = Path(__file__).parent.parent
        presets_dir = repo_root / "configs" / "presets"
        if presets_dir.exists():
            for preset in sorted(presets_dir.glob("*.yaml")):
                print(f"  - {preset.stem}")
        sys.exit(1)

    arg = sys.argv[1]
    repo_root = Path(__file__).parent.parent
    target = repo_root / "config.yaml"

    # Check if it's a preset name or a path
    if "/" in arg or "\\" in arg or Path(arg).exists():
        source = Path(arg)
    else:
        source = repo_root / "configs" / "presets" / f"{arg}.yaml"

    if not source.exists():
        print(f"Error: Configuration not found: {source}")
        print("\nAvailable presets:")
        presets_dir = repo_root / "configs" / "presets"
        if presets_dir.exists():
            for preset in sorted(presets_dir.glob("*.yaml")):
                print(f"  - {preset.stem}")
        sys.exit(1)

    # Backup current config
    if target.exists():
        backup = repo_root / "config.yaml.backup"
        shutil.copy2(target, backup)
        print(f"✓ Backed up current config to: {backup.name}")

    # Copy new config
    shutil.copy2(source, target)
    print(f"✓ Activated configuration: {source.name}")
    print(f"  Config file: {target}")

    # Show what changed
    print("\nActive configuration:")
    with open(target) as f:
        lines = f.readlines()[:10]
        for line in lines:
            if line.strip() and not line.strip().startswith('#'):
                print(f"  {line.rstrip()}")
        if len(lines) == 10:
            print("  ...")


if __name__ == "__main__":
    main()