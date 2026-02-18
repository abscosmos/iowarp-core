#!/bin/bash
# Shared devcontainer post-start setup
# Called every time the container starts (including restarts).
# Dynamically matches the docker socket GID so the dev user can
# access Docker regardless of the host's docker group ID.

DEVUSER="iowarp"

if [ -S /var/run/docker.sock ]; then
    SOCK_GID=$(stat -c '%g' /var/run/docker.sock)
    if ! id -G "${DEVUSER}" | tr ' ' '\n' | grep -qx "${SOCK_GID}"; then
        getent group "${SOCK_GID}" >/dev/null 2>&1 \
            || sudo groupadd -g "${SOCK_GID}" docker_host
        sudo usermod -aG "${SOCK_GID}" "${DEVUSER}"
    fi
    sudo chmod 660 /var/run/docker.sock
fi
