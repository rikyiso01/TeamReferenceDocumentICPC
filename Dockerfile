FROM nixos/nix

WORKDIR /app
COPY ./flake.nix ./
COPY ./package.json ./pnpm-lock.yaml ./
RUN nix develop pnpm install
COPY reference.md ./
RUN nix develop pnpm compile
