#!/usr/bin/env bash

# setup-zig-linker.sh
# Sets up Zig as linker for Linux cross-compilation targets

if [[ "$1" == *"linux"* ]]; then
  RUST_TARGET="$2"

  mkdir -p $HOME/.cargo/bin
  cat << EOF > $HOME/.cargo/bin/${RUST_TARGET}-cc
#!/usr/bin/env bash
exec zig cc -target ${RUST_TARGET} "\$@"
EOF

  chmod +x $HOME/.cargo/bin/${RUST_TARGET}-cc

  RUST_TARGET_UPPER=$(echo "$RUST_TARGET" | tr '[:lower:]' '[:upper:]')
  RUST_TARGET_NO_DASH=$(echo "$RUST_TARGET_UPPER" | tr '-' '_')

  echo "CARGO_TARGET_${RUST_TARGET_NO_DASH}_LINKER=$HOME/.cargo/bin/$RUST_TARGET-cc"
  echo "PATH=$HOME/.cargo/bin:$PATH"
fi
