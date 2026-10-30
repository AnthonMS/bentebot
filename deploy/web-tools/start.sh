#!/bin/sh
set -eu
# Used only to sign SearXNG preferences. It is not an API credential and is never logged.
SEARXNG_SECRET="$(head -c 32 /dev/urandom | base64)"
export SEARXNG_SECRET
exec /usr/local/searxng/entrypoint.sh
