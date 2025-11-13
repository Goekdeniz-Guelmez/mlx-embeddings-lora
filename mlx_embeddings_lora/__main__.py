import importlib
import sys

if __name__ == "__main__":
    subcommands = ["train"]
    
    if len(sys.argv) < 2:
        raise ValueError(f"CLI requires a subcommand in {subcommands}")
    
    subcommand = sys.argv[1]
    
    if subcommand in subcommands:
        submodule = importlib.import_module(f"mlx_embeddings_lora.{subcommand}")
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        submodule.main()
    elif subcommand == "--version":
        from mlx_embeddings_lora import __version__
        print(__version__)
    else:
        raise ValueError(f"CLI requires a subcommand in {subcommands}")
    




# Copyright © 2025 Apple Inc.

import importlib
import sys

if __name__ == "__main__":
    subcommands = {
        "train",
    }
    if len(sys.argv) < 2:
        raise ValueError(f"CLI requires a subcommand in {subcommands}")
    subcommand = sys.argv.pop(1)
    if subcommand in subcommands:
        submodule = importlib.import_module(f"mlx_embeddings_lora.{subcommand}")
        submodule.main()
    elif subcommand == "--version":
        from mlx_embeddings_lora import __version__

        print(__version__)
    else:
        raise ValueError(f"CLI requires a subcommand in {subcommands}")
