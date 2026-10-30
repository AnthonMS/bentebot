#!/bin/bash
set -euo pipefail
umask 077
viewer_secret_dir="${BENTEBOT_VIEWER_SECRETS_DIR:?Set an absolute private secrets directory}"
mkdir -p -- "$viewer_secret_dir"
IFS= read -r -s -p 'Paste the NEW viewer account token (hidden), then press Enter: ' viewer_credential
printf '\n'
if [[ ${#viewer_credential} -lt 32 || "$viewer_credential" == Bot\ * ]]; then
    unset viewer_credential
    printf 'Invalid credential format. Use the new user account token, without a Bot prefix.\n' >&2
    exit 1
fi
if [[ ! -s "$viewer_secret_dir/control-key" ]]; then
    openssl rand -hex 32 > "$viewer_secret_dir/control-key"
fi
printf '%s\n' "$viewer_credential" > "$viewer_secret_dir/discord-token"
unset viewer_credential
chown 99:100 -- "$viewer_secret_dir" "$viewer_secret_dir/control-key" "$viewer_secret_dir/discord-token"
chmod 700 -- "$viewer_secret_dir"
chmod 400 -- "$viewer_secret_dir/control-key" "$viewer_secret_dir/discord-token"
printf 'Viewer secrets saved. No credential was printed. The service has not been started.\n'
