#!/bin/sh
# Reuse the established head artwork so the reviewed leg revision cannot change the palette.
set -eu
cd "$(dirname "$0")/.."
color_tmp=$(mktemp -d /tmp/girl-run-color-match.XXXXXX)
out=docs/quality/run-thigh-v3-colormatch
mkdir -p "$out"
magick docs/quality/run-thigh-v3/cycle.png +repage -crop 640x724 +repage "$color_tmp/frame-%d.png"
# Isolate the original first-frame head above the shoulder/neck boundary.
magick -size 640x724 xc:black -fill white -stroke none \
    -draw 'polygon 0,0 640,0 640,390 502,390 481,403 446,403 403,396 375,393 350,394 328,408 305,420 274,426 209,426 194,451 169,483 0,495' \
    -alpha off "$color_tmp/head-source-mask.png"
# The revised frame's head occupies a slightly different silhouette. Clear that
# full old silhouette while retaining the near arm, shoulder, torso and both legs.
magick -size 640x724 xc:black -fill white -stroke none \
    -draw 'polygon 0,0 640,0 640,393 490,393 477,418 451,428 437,412 403,405 369,396 344,400 328,411 308,420 282,434 259,444 229,449 212,478 185,504 0,506' \
    -alpha off "$color_tmp/head-old-mask.png"
# Multiply the mask by source alpha; otherwise the cutout would contain opaque empty pixels.
magick "$color_tmp/frame-0.png" -alpha extract "$color_tmp/source-alpha.png"
magick "$color_tmp/source-alpha.png" "$color_tmp/head-source-mask.png" \
    -compose Multiply -composite "$color_tmp/head-alpha.png"
magick "$color_tmp/frame-0.png" "$color_tmp/head-alpha.png" \
    -alpha off -compose CopyOpacity -composite "$color_tmp/head.png"
magick "$color_tmp/head-old-mask.png" -alpha off -negate "$color_tmp/body-mask.png"
magick "$color_tmp/frame-2.png" -alpha extract "$color_tmp/target-alpha.png"
magick "$color_tmp/target-alpha.png" "$color_tmp/body-mask.png" \
    -compose Multiply -composite "$color_tmp/body-alpha.png"
magick "$color_tmp/frame-2.png" "$color_tmp/body-alpha.png" \
    -alpha off -compose CopyOpacity -composite "$color_tmp/body.png"
magick "$color_tmp/body.png" "$color_tmp/head.png" -geometry +0+6 -compose Over -composite \
    "$color_tmp/frame-2.png"
cp "$color_tmp/frame-2.png" "$out/frame3.png"
magick "$color_tmp/frame-0.png" "$color_tmp/frame-1.png" \
    "$color_tmp/frame-2.png" "$color_tmp/frame-3.png" +append "$out/cycle.png"
magick -delay 32 "$color_tmp/frame-0.png" "$color_tmp/frame-1.png" \
    "$color_tmp/frame-2.png" "$color_tmp/frame-3.png" \
    -background '#fff4ec' -alpha remove -alpha off -resize 75% -loop 0 "$out/run-slow.gif"
magick "$out/cycle.png" -background '#fff4ec' -alpha remove -alpha off \
    -resize 60% "$out/contact.png"
printf 'Color-matched preview: %s\nIntermediates: %s\n' "$out/run-slow.gif" "$color_tmp"
