#!/usr/bin/env bash
set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run this script with sudo:"
  echo "  sudo bash $0"
  exit 1
fi

echo "[1/4] Collecting currently held NVIDIA-related packages"
mapfile -t held_nvidia_pkgs < <(apt-mark showhold | grep -E '^((lib)?nvidia|xserver-xorg-video-nvidia|nvidia-)' || true)

if [[ "${#held_nvidia_pkgs[@]}" -gt 0 ]]; then
  echo "[2/4] Removing the oversized NVIDIA hold list"
  apt-mark unhold "${held_nvidia_pkgs[@]}" || true
else
  echo "[2/4] No NVIDIA-related holds were present"
fi

echo "[3/4] Selecting only the installed 535 driver stack to keep locked"
driver_regex='^(nvidia-driver-535|nvidia-dkms-535|nvidia-kernel-common-535|nvidia-kernel-source-535|nvidia-compute-utils-535|nvidia-utils-535|xserver-xorg-video-nvidia-535|nvidia-firmware-535-535\.288\.01|libnvidia-cfg1-535(:amd64|:i386)?|libnvidia-common-535|libnvidia-compute-535(:amd64|:i386)?|libnvidia-decode-535(:amd64|:i386)?|libnvidia-encode-535(:amd64|:i386)?|libnvidia-extra-535(:amd64|:i386)?|libnvidia-fbc1-535(:amd64|:i386)?|libnvidia-gl-535(:amd64|:i386)?)$'
mapfile -t installed_driver_pkgs < <(dpkg-query -W -f='${db:Status-Abbrev} ${binary:Package}\n' 'nvidia-*' 'libnvidia-*' 'xserver-xorg-video-nvidia-*' 2>/dev/null | grep -E '^(ii|hi) ' | awk '{print $2}' | grep -E "${driver_regex}" | sort -u)

if [[ "${#installed_driver_pkgs[@]}" -eq 0 ]]; then
  echo "Could not determine the installed NVIDIA 535 driver package set."
  exit 1
fi

echo "[4/4] Holding only the active 535 driver packages"
apt-mark hold "${installed_driver_pkgs[@]}"

echo
echo "Final NVIDIA hold list:"
apt-mark showhold | grep -E '^((lib)?nvidia|xserver-xorg-video-nvidia|nvidia-)' | sort -u
