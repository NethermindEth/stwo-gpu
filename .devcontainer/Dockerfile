FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

RUN <<EOF
  apt-get update
  export DEBIAN_FRONTEND=noninteractive
  apt-get -y install --no-install-recommends git curl cmake ssh
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
EOF