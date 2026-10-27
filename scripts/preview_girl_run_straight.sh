#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
old=art/stage/run_inbetweens_v1/frames
asset=art/stage/run_inbetweens_v2
out=docs/quality/run-eight-frame-v2
scratch=$(mktemp -d /tmp/girl-run-straight.XXXXXX)
mkdir -p "$asset/frames" "$out"
for i in 0 1 2 3 4 5 6 7; do cp "$old/frame-$i.png" "$asset/frames/frame-$i.png"; done
# Keep the original upper body pixels, including face, hair and arms.
# The generated lower crop changes both legs; its skirt hem covers the join.
magick "$asset/body-straight.png" +repage -resize '640x349!' -crop 640x204+0+145 +repage "$scratch/lower.png"
magick "$old/frame-3.png" +repage -crop 640x520+0+0 +repage "$scratch/upper.png"
magick "$scratch/upper.png" "$scratch/lower.png" -append "$asset/frames/frame-3.png"
cp "$asset/frames/frame-3.png" "$asset/straight-pose.png"
magick "$asset"/frames/frame-*.png +append "$asset/cycle-eight.png"
magick -delay 8 "$asset"/frames/frame-*.png -background '#fff4ec' -alpha remove -alpha off -resize 75% -loop 0 "$out/run.gif"
magick -delay 16 "$asset"/frames/frame-*.png -background '#fff4ec' -alpha remove -alpha off -resize 75% -loop 0 "$out/run-slow.gif"
magick montage -font /System/Library/Fonts/Supplemental/Arial.ttf -pointsize 22 \
    -label 'Key 2' "$asset/frames/frame-2.png" \
    -label 'Straight-leg transition' "$asset/frames/frame-3.png" \
    -label 'Key 3' "$asset/frames/frame-4.png" \
    -background '#fff4ec' -fill '#603f34' -tile 3x1 -geometry 320x362+8+8 "$out/comparison.png"
