{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-23.11";
    nixpkgs-unstable.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = inputs:
    inputs.flake-utils.lib.eachSystem ["x86_64-linux"] (system: let
      config = {
        allowUnfree = true;
        # cudaSupport = true;
      };

      overlay-unstable = final: prev: {
        unstable = import inputs.nixpkgs-unstable {
          inherit system config;
        };
      };
      pkgs = import inputs.nixpkgs {
        inherit system config;
        overlays = [
          overlay-unstable
        ];
      };

      fhs = pkgs.buildFHSEnv {
        name = "fhs-shell";
        targetPkgs = p: (packages p) ++ custom-commands;
        runScript = "${pkgs.zsh}/bin/zsh";
        profile = ''
          source ./.venv/bin/activate
          source .env
        '';
      };
      pip-install = pkgs.buildFHSEnv {
        name = "pip-install";
        targetPkgs = packages;
        runScript = ''
          #!/usr/bin/env bash
          source ./.venv/bin/activate
          pip install -r ./requirements.txt
        '';
      };
      chatmed-serve = pkgs.buildFHSEnv {
        name = "chatmed-serve";
        targetPkgs = packages;
        runScript = ''
          #!/usr/bin/env bash
          source ./.venv/bin/activate

          ${pkgs.redis}/bin/redis-server &
          api-serve &
          streamlit run ./src/app.py
        '';
      };
      api-serve = pkgs.buildFHSEnv {
        name = "api-serve";
        targetPkgs = packages;
        runScript = ''
          #!/usr/bin/env bash
          source ./.venv/bin/activate

          cd src
          uvicorn api_main:app --reload
        '';
      };

      custom-commands = [
        pip-install
        api-serve
        chatmed-serve
      ];

      packages = pkgs: (with pkgs; [
        (pkgs.python310.withPackages (ps:
          with ps; [
          ]))
        python311Packages.python-lsp-server
        python311Packages.ruff-lsp # python linter
        python311Packages.black # python formatter

        pkgs.python310Packages.pip
        pkgs.python310Packages.virtualenv
        zlib # idk why/if this is needed

        redis

        # virtualenv .venv
        # source ./.venv/bin/activate
        # pip install ..
        # python311Packages.venvShellHook # ??
      ]);
    in {
      devShells.default = pkgs.mkShell {
        nativeBuildInputs = [fhs] ++ custom-commands ++ packages pkgs;
        shellHook = ''
          source .env
        '';
      };
      # devShells.default = fhs.env;
    });
}
