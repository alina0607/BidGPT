#!/bin/sh
# Pack generated body in-betweens with the original, color-locked head.
set -eu
cd "$(dirname "$0")/.."
asset=art/stage/run_inbetweens_v1
out=docs/quality/run-eight-frame-v1
scratch=$(mktemp -d /tmp/girl-run-eight.XXXXXX)
mkdir -p "$out" "$asset/frames"
magick -size 640x724 xc:black -fill white -stroke none \
    -draw 'polygon 0,0 640,0 640,390 502,390 481,403 446,403 403,396 375,393 350,394 328,408 305,420 274,426 209,426 194,451 169,483 0,495' \
    -alpha off "$scratch/head-mask.png"
magick "$asset/references/key-0.png" -alpha extract "$scratch/head-a.png"
magick "$scratch/head-a.png" "$scratch/head-mask.png" -compose Multiply -composite "$scratch/head-a.png"
magick "$asset/references/key-0.png" "$scratch/head-a.png" -alpha off -compose CopyOpacity -composite "$scratch/head.png"
# Clear generated partial chin/hair; retain only neck, body, arms and legs.
magick -size 640x724 xc:white -fill black -stroke none \
    -draw 'polygon 0,0 640,0 640,394 500,394 473,406 445,403 410,399 375,395 348,401 330,419 302,433 270,440 225,440 208,488 175,503 0,505' \
    -alpha off "$scratch/body-mask.png"
for i in 0 1 2 3; do
    even=$((i * 2))
    odd=$((even + 1))
    cp "$asset/references/key-$i.png" "$asset/frames/frame-$even.png"
    body_source="$asset/body-mid-$i.png"
    if [ "$i" = 1 ] || [ "$i" = 3 ]; then body_source="$asset/body-mid-$i-revised.png"; fi
    magick "$body_source" +repage -resize '640x349!' \
        -background none -gravity south -extent 640x724 "$scratch/body-$i.png"
    magick "$scratch/body-$i.png" -alpha extract "$scratch/a.png"
    magick "$scratch/a.png" "$scratch/body-mask.png" -compose Multiply -composite "$scratch/a.png"
    magick "$scratch/body-$i.png" "$scratch/a.png" -alpha off -compose CopyOpacity -composite "$scratch/body-$i.png"
    magick "$scratch/body-$i.png" "$scratch/head.png" -geometry +0+3 -compose Over -composite "$asset/frames/frame-$odd.png"
done
magick "$asset"/frames/frame-*.png +append "$asset/cycle-eight.png"
magick -delay 8 "$asset"/frames/frame-*.png -background '#fff4ec' -alpha remove -alpha off -resize 75% -loop 0 "$out/run.gif"
magick -delay 16 "$asset"/frames/frame-*.png -background '#fff4ec' -alpha remove -alpha off -resize 75% -loop 0 "$out/run-slow.gif"
magick montage -font /System/Library/Fonts/Supplemental/Arial.ttf "$asset"/frames/frame-*.png -background '#fff4ec' -tile 4x2 -geometry 320x362+8+8 "$out/contact.png"
for i in 0 1 2 3; do
    a=$((i * 2)); b=$((a + 1)); c=$(((a + 2) % 8))
    magick montage -font /System/Library/Fonts/Supplemental/Arial.ttf -pointsize 22 \
        -label "Key $((i+1))" "$asset/frames/frame-$a.png" \
        -label "NEW $((i+1)) -> $(((i+1)%4+1))" "$asset/frames/frame-$b.png" \
        -label "Key $(((i+1)%4+1))" "$asset/frames/frame-$c.png" \
        -background '#fff4ec' -fill '#603f34' -tile 3x1 -geometry 280x317+8+8 "$out/pair-$i.png"
done
magick "$out"/pair-*.png -append "$out/pair-comparison.png"
printf 'Preview: %s\n' "$out/run.gif"
