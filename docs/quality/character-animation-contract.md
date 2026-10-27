# Character animation acceptance contract

The v3 strip warp / two-pose dissolve was rejected by the owner. It was the wrong approach
for these painted full-frame sprites; numerical eye alignment did not establish visual quality.

## Current safe runtime

- One full sprite exposure per character per render. No horizontal strip deformation,
  independent head scaling, body-part patchwork or two-character alpha dissolves.
- Original idle and approved run PNGs are unchanged. Idle plays its authored frames at a
  fixed root and scale; procedural breathing deformation is removed.
- Releasing movement continues the existing stride to its planted exposure before idle.
- The rejected turn atlas is no longer loaded. Turning is temporarily disabled: a direct
  whole-frame switch is a known limitation, not a completed smooth transition.
- `art/stage/girl_turn_v4` is a rejected art candidate. It is NOT installed as runtime art.
- `art/stage/girl_turn_v5` was also rejected by the owner for style, proportions and colour
  differences from idle. Do not reuse its generated drawings as art/model references.
  The v6 restart must derive directly from original runtime idle, not from rejected artwork.

## Required before turning can be re-enabled

1. Draw complete coherent exposures between the actual idle and actual run-ready endpoints.
   Neck/shoulders/hips remain connected; whole body turns together, with purposeful hair lag.
2. Fixed canvas, pivot, floor baseline and camera scale. Preserve head/leg/boot proportions and
   the approved palette/brushwork. Do not independently normalize body parts to hide bad art.
3. Explicit per-frame timings and state transitions; step through one sprite frame at a time.
4. Test starting, releasing at each gait phase, brief taps, both directions and re-pressing while
   stopping. The actual rendered exposure is the authority, not only state/coverage tests.
5. Inspect actual game recordings at normal speed and enlarged frame contacts; reject visible
   disconnections, size/colour flashes, doubled limbs, moving pivots or clipping before delivery.

Frame-based sprite playback is a standard supported approach, not the only valid 2D animation
technique. A properly authored skeletal rig can also work, but no such rig exists for these
flattened sprites. Reference: [Godot's 2D sprite animation documentation](https://docs.godotengine.org/en/stable/tutorials/2d/2d_sprite_animation.html).

## Owner-approved original-pixel restart

The owner subsequently approved original-pixel layered authoring, manual key poses,
and repainting only missing views. `art/stage/girl_turn_rig_v1` starts this work.
Its connected UV mesh and source-region extraction are construction tools, not a
finished skeletal rig or an exception to the visual acceptance requirements.
Continuous surface interpolation is allowed for small motions; horizontal strips,
disconnected independently scaled parts and whole-pose dissolves remain rejected.
The small-angle engine probe is NOT a complete-turn test and must never be installed
as `girl_turn_approved`. Final GIF delivery still requires the whole transition chain.
