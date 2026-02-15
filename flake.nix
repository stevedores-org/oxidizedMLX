{
  description = "oxidizedMLX — Rust-first MLX runtime";

  nixConfig = {
    extra-substituters = [ "https://nix-cache.stevedores.org/oxidizedmlx" ];
    extra-trusted-public-keys = [ "oxidizedmlx-cache-1:uG3uzexkJno1b3b+dek7tHnHzr1p6MHxIoVTqnp/JBI=" ];
  };

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

    crane.url = "github:ipetkov/crane";

    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, crane, rust-overlay, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ (import rust-overlay) ];
        };

        rustToolchain = pkgs.rust-bin.stable.latest.default.override {
          extensions = [ "rustfmt" "clippy" ];
        };

        craneLib = (crane.mkLib pkgs).overrideToolchain rustToolchain;

        # Source filtering: only include Rust-relevant files
        src = pkgs.lib.cleanSourceWith {
          src = ./.;
          filter = path: type:
            (craneLib.filterCargoSources path type)
            || pkgs.lib.hasSuffix ".h" path;
        };

        commonArgs = {
          inherit src;
          strictDeps = true;

          buildInputs = pkgs.lib.optionals pkgs.stdenv.isDarwin [
            pkgs.libiconv
            pkgs.darwin.apple_sdk.frameworks.Metal
            pkgs.darwin.apple_sdk.frameworks.Foundation
          ];

          nativeBuildInputs = [
            pkgs.pkg-config
          ];
        };

        # Build dependencies once (shared across all crate builds)
        cargoArtifacts = craneLib.buildDepsOnly (commonArgs // {
          # Exclude mlx-sys from dep build (requires MLX_SRC for cpp feature)
          cargoExtraArgs = "--workspace --exclude mlx-sys";
        });

        # Full workspace build (excludes mlx-sys which needs external C++ lib)
        workspaceBuild = craneLib.buildPackage (commonArgs // {
          inherit cargoArtifacts;
          pname = "oxidizedmlx";
          version = "0.1.0";
          cargoExtraArgs = "--workspace --exclude mlx-sys";
        });

        # Workspace clippy check
        workspaceClippy = craneLib.cargoClippy (commonArgs // {
          inherit cargoArtifacts;
          cargoClippyExtraArgs = "--workspace --exclude mlx-sys --all-targets -- -D warnings";
        });

        # Workspace tests
        workspaceTests = craneLib.cargoNextest (commonArgs // {
          inherit cargoArtifacts;
          cargoNextestExtraArgs = "--workspace --exclude mlx-sys";
        });

        # Format check
        workspaceFmt = craneLib.cargoFmt {
          inherit src;
        };

        # CLI binary
        mlxCli = craneLib.buildPackage (commonArgs // {
          inherit cargoArtifacts;
          pname = "mlx-cli";
          version = "0.1.0";
          cargoExtraArgs = "-p mlx-cli";
        });

      in {
        checks = {
          inherit workspaceBuild workspaceClippy workspaceTests workspaceFmt;
        };

        packages = {
          default = workspaceBuild;
          cli = mlxCli;
        };

        devShells.default = craneLib.devShell {
          checks = self.checks.${system};

          packages = with pkgs; [
            attic-client
            cargo-nextest
            just
          ];
        };
      }
    );
}
