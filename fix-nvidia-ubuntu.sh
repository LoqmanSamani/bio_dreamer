#!/usr/bin/env bash
set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run this script with sudo:"
  echo "  sudo bash $0"
  exit 1
fi

CUDA_LIST="/etc/apt/sources.list.d/cuda-ubuntu2404-x86_64.list"
PPA_SOURCE="/etc/apt/sources.list.d/graphics-drivers-ubuntu-ppa-noble.sources"

echo "[1/7] Disabling conflicting NVIDIA package sources"
if [[ -f "${CUDA_LIST}" ]]; then
  mv "${CUDA_LIST}" "${CUDA_LIST}.disabled"
fi
if [[ -f "${PPA_SOURCE}" ]]; then
  mv "${PPA_SOURCE}" "${PPA_SOURCE}.disabled"
fi

echo "[2/7] Refreshing apt metadata"
apt-get update

echo "[3/7] Temporarily unholding NVIDIA packages so apt can repair them"
mapfile -t held_nvidia_pkgs < <(apt-mark showhold | grep -E '^((lib)?nvidia|xserver-xorg-video-nvidia)' || true)
if [[ "${#held_nvidia_pkgs[@]}" -gt 0 ]]; then
  apt-mark unhold "${held_nvidia_pkgs[@]}" || true
fi

echo "[4/7] Removing mismatched NVIDIA helper packages from external repos"
DEBIAN_FRONTEND=noninteractive apt-get -y --allow-change-held-packages purge \
  libnvidia-cfg1 \
  libnvidia-compute \
  libnvidia-gpucomp \
  nvidia-persistenced \
  nvidia-settings || true

echo "[5/7] Repairing package state and reinstalling the Ubuntu stable NVIDIA 535 driver"
dpkg --remove --force-remove-reinstreq libnvidia-cfg1:amd64 2>/dev/null || true
DEBIAN_FRONTEND=noninteractive apt-get -y --allow-change-held-packages install \
  libnvidia-cfg1-535 \
  nvidia-driver-535 || true
DEBIAN_FRONTEND=noninteractive apt-get -y --allow-change-held-packages --fix-broken install
dpkg --configure -a
DEBIAN_FRONTEND=noninteractive apt-get -y --allow-change-held-packages install \
  --reinstall \
  --no-install-recommends \
  libnvidia-cfg1-535 \
  nvidia-driver-535

echo "[6/7] Holding the currently installed NVIDIA driver packages"
mapfile -t installed_pkgs < <(dpkg-query -W -f='${binary:Package}\n' 'nvidia-*' 'libnvidia-*' 'xserver-xorg-video-nvidia-*' 2>/dev/null | sort -u)
if [[ "${#installed_pkgs[@]}" -gt 0 ]]; then
  apt-mark hold "${installed_pkgs[@]}"
fi

echo "[7/7] Verifying driver status"
modprobe nvidia || true
echo
echo "Held NVIDIA packages:"
apt-mark showhold | grep -E '^((lib)?nvidia|xserver-xorg-video-nvidia)' || true
echo
echo "nvidia-smi output:"
nvidia-smi || true
echo
echo "Kernel modules:"
lsmod | grep -E 'nvidia|nouveau' || true

cat <<'EOF'

If nvidia-smi still fails, reboot once so the rebuilt kernel module is loaded:
  sudo reboot

If you later need CUDA repo updates again, re-enable its list file manually:
  sudo mv /etc/apt/sources.list.d/cuda-ubuntu2404-x86_64.list.disabled \
          /etc/apt/sources.list.d/cuda-ubuntu2404-x86_64.list

EOF
