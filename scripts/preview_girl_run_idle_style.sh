#!/bin/sh
# Whole-sprite registration only: no recoloring, masks or head/body compositing.
set -eu
cd "$(dirname "$0")/.."
asset=art/stage/run_idle_style_v1
out=docs/quality/run-idle-style-v1
scratch=$(mktemp -d /tmp/run-idle-style.XXXXXX)
mkdir -p "$asset/frames" "$out"
magick "$asset/raw/frame-0-revised.png" +repage -resize 556x "$scratch/resized-0.png"
magick -size 640x724 xc:none "$scratch/resized-0.png" -geometry +7+76 -compose Over -composite "$asset/frames/frame-0.png"
magick "$asset/raw/frame-1.png" +repage -resize 574x "$scratch/resized-1.png"
magick -size 640x724 xc:none "$scratch/resized-1.png" -geometry -10+69 -compose Over -composite "$asset/frames/frame-1.png"
magick "$asset/raw/frame-2.png" +repage -resize 568x "$scratch/resized-2.png"
magick -size 640x724 xc:none "$scratch/resized-2.png" -geometry -4+69 -compose Over -composite "$asset/frames/frame-2.png"
magick "$asset/raw/frame-3.png" +repage -resize 594x "$scratch/resized-3.png"
magick -size 640x724 xc:none "$scratch/resized-3.png" -geometry -2+33 -compose Over -composite "$asset/frames/frame-3.png"
magick "$asset/raw/frame-4.png" +repage -resize 555x "$scratch/resized-4.png"
magick -size 640x724 xc:none "$scratch/resized-4.png" -geometry +8+99 -compose Over -composite "$asset/frames/frame-4.png"
magick "$asset/raw/frame-5.png" +repage -resize 547x "$scratch/resized-5.png"
magick -size 640x724 xc:none "$scratch/resized-5.png" -geometry +5+88 -compose Over -composite "$asset/frames/frame-5.png"
magick "$asset/raw/frame-6.png" +repage -resize 581x "$scratch/resized-6.png"
magick -size 640x724 xc:none "$scratch/resized-6.png" -geometry -17+70 -compose Over -composite "$asset/frames/frame-6.png"
magick "$asset/raw/frame-7.png" +repage -resize 577x "$scratch/resized-7.png"
magick -size 640x724 xc:none "$scratch/resized-7.png" -geometry -2+67 -compose Over -composite "$asset/frames/frame-7.png"
magick "$asset"/frames/frame-*.png +append "$asset/cycle-eight.png"
magick -delay 8 "$asset"/frames/frame-*.png -background '#fff4ec' -alpha remove -alpha off -resize 75% -loop 0 "$out/run.gif"
magick -delay 16 "$asset"/frames/frame-*.png -background '#fff4ec' -alpha remove -alpha off -resize 75% -loop 0 "$out/run-slow.gif"
magick montage -font /System/Library/Fonts/Supplemental/Arial.ttf -pointsize 18 \
-label "Frame 1" "$asset/frames/frame-0.png" -label "Frame 2" "$asset/frames/frame-1.png" -label "Frame 3" "$asset/frames/frame-2.png" -label "Frame 4" "$asset/frames/frame-3.png" -label "Frame 5" "$asset/frames/frame-4.png" -label "Frame 6" "$asset/frames/frame-5.png" -label "Frame 7" "$asset/frames/frame-6.png" -label "Frame 8" "$asset/frames/frame-7.png" \
 -background '#fff4ec' -fill '#603f34' -tile 4x2 -geometry 280x317+8+8 "$out/contact.png"
magick "$asset/references/idle-game.png" -resize x540 -background none -gravity center -extent 640x724 "$scratch/idle.png"
magick montage -font /System/Library/Fonts/Supplemental/Arial.ttf -pointsize 18 \
 -label Idle "$scratch/idle.png" -label 'Key 2' "$asset/frames/frame-2.png" \
 -label 'Straight-leg transition' "$asset/frames/frame-3.png" -label 'Key 3' "$asset/frames/frame-4.png" \
 -background '#fff4ec' -fill '#603f34' -tile 4x1 -geometry 280x317+8+8 "$out/idle-comparison.png"

