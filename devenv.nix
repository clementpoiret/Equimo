{ pkgs, lib, ... }:
let
  buildInputs = with pkgs; [
    stdenv.cc.cc
    libuv
    zlib
  ];
in
{
  packages = with pkgs; [ python312 ];

  env = {
    # UV_PYTHON = "${pkgs.python312}/bin/python";
    LD_LIBRARY_PATH = lib.makeLibraryPath buildInputs;
  };

  languages.python = {
    enable = true;
    package = pkgs.python312;
    uv = {
      enable = true;
      sync.enable = true;
    };
  };

  # enterShell = "";
  enterTest = "uv run pytest";
}
