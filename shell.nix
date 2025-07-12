{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = [
    pkgs.python3
    pkgs.python3Packages.pip
    pkgs.python3Packages.virtualenv
  ];

  shellHook = ''
    if [ ! -d .venv ]; then
      python -m venv .venv
      . .venv/bin/activate
      pip install -r requirements.txt
    else
      . .venv/bin/activate
    fi
  '';
}
