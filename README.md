# Nix Python Devenv

A Python development environment template using [devenv](https://devenv.sh) with working C bindings,
managed by [uv](https://docs.astral.sh/uv/) package manager.

A version of this repo with CUDA is available on the
[cuda branch](https://github.com/clementpoiret/nix-python-devenv/tree/cuda).

## Features

- CUDA toolkit and CUDNN support (see the `cuda` branch)
- Python 3.x environment
- Fast package management with uv
- Automatic environment activation with direnv
- Example script using numpy

## Installation

1. Install [Nix](https://nixos.org/download/):
   ```bash
   sh <(curl -L https://nixos.org/nix/install) --daemon
   ```

2. Install [devenv](https://devenv.sh/)
   ```bash
   nix-env -iA devenv -f https://github.com/NixOS/nixpkgs/tarball/nixpkgs-unstable
   ```

3. Install [direnv](https://direnv.net/) (optional but recommended)

4. Clone and setup:
   ```bash
   git clone --single-branch --branch cuda git@github.com:clementpoiret/nix-python-devenv.git

   # Allow direnv to manage the environment
   direnv allow
   ```

## Usage

The environment automatically:
- Activates the Python virtual environment
- Sets up LD_LIBRARY_PATH
- Provides the `hello` command to test C Bindings

Run the sample script:
```bash
hello
```

This will display the output of numpy functions.
