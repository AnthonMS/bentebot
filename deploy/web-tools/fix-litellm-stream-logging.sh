#!/bin/sh
# Repair only the reviewed served-output logging helper in the current container.
set -eu
cd /mnt/user/appdata/bentebot
stamp=$(date -u +%Y%m%dT%H%M%SZ)
out="artifacts/litellm-index-guard-$stamp"
mkdir -p "$out"
patch=scripts/patch_litellm_served_output.py

printf 'Validate installed source and run logging regression cases\n'
docker exec -i LiteLLM python - < "$patch"
target=$(docker exec LiteLLM python -c 'import importlib; print(importlib.import_module("litellm.litellm_core_utils.served_output_texts").__file__)')
case "$target" in
  /*/litellm/litellm_core_utils/served_output_texts.py) ;;
  *) printf 'Unexpected installed module path; stopping\n' >&2; exit 1 ;;
esac
docker inspect --format '{{.Config.Image}} {{.Image}}' LiteLLM > "$out/container-image.txt"
docker exec LiteLLM python -c 'import importlib.metadata; print(importlib.metadata.version("litellm"))' > "$out/version.txt"
docker cp "LiteLLM:$target" "$out/served_output_texts.before.py"
printf '%s\n' "$target" > "$out/module-path.txt"
printf 'Apply guarded logging helper\n'
docker exec -i LiteLLM python - --apply < "$patch" > "$out/apply.txt"
cat "$out/apply.txt"
# Verify the file in a fresh Python process before restarting the proxy.
docker exec -i LiteLLM python - < "$patch"
printf 'Restart LiteLLM to load the repaired helper (brief gateway reconnect)\n'
docker restart LiteLLM
printf 'LITELLM_INDEX_GUARD_APPLIED\nBackup: /mnt/user/appdata/bentebot/%s\n' "$out"
printf 'This hotfix survives restart, but must be reapplied after container recreation or an image update.\n'
