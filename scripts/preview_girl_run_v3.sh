#!/bin/sh
# Encode the reviewed thigh revision into a GIF without changing runtime art.
set -eu
cd "$(dirname "$0")/.."
preview_tmp=$(mktemp -d /tmp/girl-run-v3-preview.XXXXXX)
out=docs/quality/run-thigh-v3
mkdir -p "$out"
magick art/stage/girl_run_cycle_v5.png -crop 640x724 +repage "$preview_tmp/frame-%d.png"
# Uniform 0.51 scale matches the existing head dimensions and foot baseline.
magick art/stage/run_sources/frame3_hip_inward_clockwise_v3.png \
    -resize 51% "$preview_tmp/revision.png"
magick -size 640x724 xc:none "$preview_tmp/revision.png" -geometry +27+0 \
    -compose Over -composite "$preview_tmp/frame-2.png"
magick "$preview_tmp/frame-0.png" "$preview_tmp/frame-1.png" \
    "$preview_tmp/frame-2.png" "$preview_tmp/frame-3.png" +append "$out/cycle.png"
magick -delay 16 "$preview_tmp/frame-0.png" "$preview_tmp/frame-1.png" \
    "$preview_tmp/frame-2.png" "$preview_tmp/frame-3.png" \
    -background '#fff4ec' -alpha remove -alpha off -resize 75% \
    -loop 0 -layers Optimize "$out/run.gif"
magick "$out/cycle.png" -background '#fff4ec' -alpha remove -alpha off \
    -resize 60% "$out/contact.png"
printf 'GIF: %s/run.gif\n' "$out"
