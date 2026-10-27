# DGL on OpenGL

Runs programs written for the **DigiPen Graphics Library** (DGL 1.4, Direct3D 11, Windows-only) on
macOS and Linux, unchanged. Built on GLFW + OpenGL 3.3, with glad (bundled in GLFW's sources) and
`stb_image`.

This is an independent implementation of DGL's **public API** as documented on the
[DGL wiki](https://github.com/DigiPen-Faculty/DigiPen-Graphics-Library/wiki). It contains no
DigiPen code and does not need the DGL SDK.

| Piece | What it does |
|---|---|
| [`dgl/include/DGL/inc/DGL.h`](dgl/include/DGL/inc/DGL.h) | Same declarations as DGL 1.4.0 |
| [`dgl/src/DGL_GL.cpp`](dgl/src/DGL_GL.cpp) | Window, input, textures, meshes, camera, blending, render textures |
| [`win32/`](win32) | The Win32 surface DGL programs touch: types, `VK_*` codes, SAL annotations, `wWinMain → main` |
| [`cs529/`](cs529) | Optional: builds a CS529 course project (see below) |
| [`examples/dgl_demo.cpp`](examples/dgl_demo.cpp) | Self-contained demo; also built in CI |

## Behavior the documentation leaves open

These were inferred from how course code uses the API and should be compared against real DGL
on Windows:

| Behavior | This implementation | Evidence |
|---|---|---|
| Tint | `rgb = mix(rgb, tint.rgb, tint.a)` | Code uses `{0,0,0,0}` as "no tint" and `{1,0,0,0.5}` as "half red" |
| Texture shader | Ignores vertex color | Sprite meshes are built with black vertices and still show the texture |
| `DGL_Mat4` layout | Row-major `m[row][col]`, column vectors, translation in `m[0][3]`, `m[1][3]` | `Matrix2D::Translate` writes there |
| World space | Origin at window center, y up, 1 unit = 1 pixel at zoom 1 | Wiki examples |
| Camera zoom | Visible width = window width × zoom | Wiki: "smaller values move the camera in" |
| Texture coordinates | (0,0) top-left | Sprite UVs |
| `DGL_AM_MIRROR_ONCE` | Same as `MIRROR` | No core-3.3 equivalent |
| Custom HLSL shaders | Not supported; reports an error and draws with vertex colors | — |

## Testing hooks

Environment variables, read once at `DGL_System_Init`:

| Variable | Effect |
|---|---|
| `DGL_COMPAT_CAPTURE=out.png@120[,b.png@240]` | Save the frame drawn on frame N as PNG (time spent writing is excluded from the next delta time) |
| `DGL_COMPAT_EXIT_AFTER=N` | Close the window after N frames |
| `DGL_COMPAT_HOLD=0x27@30-90[,0x26@50-52]` | Hold a virtual key during frames [from, to] |

Used to verify CS529 Project 3 without a person at the keyboard: walking right at 500 px/s
measured exactly (8.33 px per 1/60 s frame), with the expected animation frames.

## Building a CS529 project

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug \
      -DSA_CS529_PROJECT="/path/to/Project3-Transformations (Student)"
cmake --build build --target run_cs529
```

- The project's `Source/` is copied into the build tree on every build; the original folder is never written.
- `LoggingSystem` is replaced with [`cs529/LoggingSystem.h`](cs529/LoggingSystem.h) (same interface),
  because the original links the Windows-only `dragon.lib`.
- The program runs from `build/compat/cs529_run/`, which links to the project's `Assets/` and `Data/`,
  so `Trace.log` is written there and not into the course folder.
- Debug builds define `_DEBUG`, so the CheatSystem scene hot-keys work as in Visual Studio.
