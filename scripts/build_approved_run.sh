#!/bin/sh
# Deterministic assembly of the owner's selected lower-body artwork, without regenerating it.
set -eu
cd "$(dirname "$0")/.."
run_tmp=$(mktemp -d /tmp/sell-anything-approved-run.XXXXXX)
src=art/stage/run_sources

magick "$src/cycle_reference.png" -crop 543x724+1086+0 +repage "$run_tmp/third.png"
magick "$src/frame3_lower_approved.png" -resize 543x304! "$run_tmp/lower.png"
# The crop was generated from y=420 of this exact reference. Transition inside
# the skirt at y=470..486; both legs and both shoes below are the approved pixels.
magick -size 543x724 xc:none "$run_tmp/lower.png" -geometry +0+420 \
    -compose Src -composite "$run_tmp/lower_full.png"
magick "$run_tmp/third.png" "$run_tmp/lower_full.png" -channel RGBA \
    -fx 'u*(1-max(0,min(1,(j-470)/16)))+v*max(0,min(1,(j-470)/16))' \
    "$run_tmp/assembled.png"
# Match the third frame's established head scale/position in v4. One uniform
# transform applies to the whole character; the head/body are never independently stretched.
magick "$run_tmp/assembled.png" -resize 621x828! "$run_tmp/scaled.png"
magick -size 640x724 xc:none "$run_tmp/scaled.png" -geometry +20-32 \
    -compose Over -composite "$run_tmp/third_aligned.png"
magick art/stage/girl_run_cycle_v4.png "$run_tmp/third_aligned.png" -geometry +1280+0 \
    -define compose:outside-overlay=false -compose Src -composite art/stage/girl_run_cycle_v5.png
magick "$run_tmp/assembled.png" docs/quality/run-approved/frame3.png
magick "$run_tmp/assembled.png" -crop 410x215+85+475 +repage \
    -background '#f6eedc' -alpha remove -alpha off \
    docs/quality/run-approved/legs.png
printf 'Approved third frame assembled; intermediates: %s\n' "$run_tmp"
