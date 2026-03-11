#!/usr/bin/env bash
set -euo pipefail

# Bootstrap environment for Trainium Kernel Fusion on Ubuntu Trn1/Trn2.
# - Installs Neuron apt repo + driver/runtime/tools
# - Creates project .venv
# - Installs Python dependencies (numpy, matplotlib, neuronx-cc)
#
# Usage:
#   bash setup_env.sh
#   bash setup_env.sh --recreate-venv
#   bash setup_env.sh --recreate-venv --prune-artifacts

RECREATE_VENV=0
PRUNE_ARTIFACTS=0
for arg in "$@"; do
  case "$arg" in
    --recreate-venv) RECREATE_VENV=1 ;;
    --prune-artifacts) PRUNE_ARTIFACTS=1 ;;
    *)
      echo "Unknown argument: $arg"
      echo "Usage: bash setup_env.sh [--recreate-venv] [--prune-artifacts]"
      exit 1
      ;;
  esac
done

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
NEURON_LIST="/etc/apt/sources.list.d/neuron.list"
NEURON_KEYRING="/etc/apt/keyrings/neuron-archive-keyring.gpg"
NEURON_LDCONF="/etc/ld.so.conf.d/neuron.conf"
PROFILE_DIR="${ROOT_DIR}/profiling"
MIN_FREE_MB=2500

free_root_mb() {
  df -Pm / | awk 'NR==2 {print $4}'
}

print_disk_state() {
  echo "[setup] Disk state:"
  df -h /
}

cleanup_caches() {
  echo "[setup] Cleaning pip/apt caches to reduce disk pressure..."
  rm -rf "${HOME}/.cache/pip" || true
  sudo apt-get clean
}

ensure_space() {
  local free_mb
  free_mb="$(free_root_mb)"
  if (( free_mb < MIN_FREE_MB )); then
    echo "[setup] WARNING: low free space on / (${free_mb} MB available)."
    echo "[setup] Recommended: >= ${MIN_FREE_MB} MB before large Neuron/pip installs."
  fi
}

echo "[setup] ROOT_DIR=${ROOT_DIR}"
print_disk_state
cleanup_caches

if [[ ${PRUNE_ARTIFACTS} -eq 1 ]]; then
  echo "[setup] Pruning profiling artifacts/reports..."
  rm -rf "${PROFILE_DIR}/artifacts" "${PROFILE_DIR}/reports"
fi

ensure_space

echo "[setup] Installing system prerequisites..."
sudo apt-get update
sudo apt-get install -y python3 python3-pip python3.12-venv curl gpg linux-headers-"$(uname -r)"

echo "[setup] Configuring AWS Neuron apt repository..."
sudo mkdir -p /etc/apt/keyrings
if [[ ! -f "${NEURON_KEYRING}" ]]; then
  curl -fsSL https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB \
    | gpg --dearmor \
    | sudo tee "${NEURON_KEYRING}" >/dev/null
fi

. /etc/os-release
echo "deb [signed-by=${NEURON_KEYRING}] https://apt.repos.neuron.amazonaws.com ${VERSION_CODENAME} main" \
  | sudo tee "${NEURON_LIST}" >/dev/null

sudo apt-get update

echo "[setup] Installing Neuron runtime/driver/tools..."
sudo apt-get install -y \
  aws-neuronx-runtime-lib \
  aws-neuronx-dkms \
  aws-neuronx-collectives \
  aws-neuronx-tools

echo "[setup] Registering Neuron shared library path..."
echo "/opt/aws/neuron/lib" | sudo tee "${NEURON_LDCONF}" >/dev/null
sudo ldconfig
cleanup_caches
ensure_space

if [[ ${RECREATE_VENV} -eq 1 && -d "${VENV_DIR}" ]]; then
  echo "[setup] Removing existing venv at ${VENV_DIR}"
  rm -rf "${VENV_DIR}"
fi

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "[setup] Creating venv..."
  python3 -m venv "${VENV_DIR}"
else
  echo "[setup] Reusing existing venv at ${VENV_DIR}"
fi

echo "[setup] Installing Python dependencies in venv..."
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
python -m pip install -U pip
python -m pip install --no-cache-dir \
  numpy \
  matplotlib \
  --extra-index-url https://pip.repos.neuron.amazonaws.com \
  "neuronx-cc==2.*"
cleanup_caches

echo "[setup] Verifying Neuron Python import..."
python - <<'PY'
from neuronxcc import nki
print("neuronxcc import OK")
PY

echo
echo "[setup] Done."
print_disk_state
echo "[setup] Activate environment:"
echo "  source \"${VENV_DIR}/bin/activate\""
echo "[setup] Suggested quick check:"
echo "  bash scripts/compile_kernels.sh --target trn2 --seqlen 512"
