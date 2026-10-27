// Exercises the DGL compat layer without any course code: a vertex-colored triangle, a textured
// quad (texture built in memory), tinting, alpha, camera movement and mouse picking.
// Arrow keys move the camera, Z/X zoom, the square follows the mouse, Esc quits.
// Set DGL_DEMO_FRAMES=N to quit after N frames (used by CI).

#include <Windows.h>
#include <sal.h>

#include <DGL.h>

#include <cstdio>
#include <cstdlib>

namespace
{

LRESULT CALLBACK WindowProc(HWND window, UINT message, WPARAM wParam, LPARAM lParam)
{
    int result = 0;
    if (DGL_System_HandleWindowsMessage(message, wParam, lParam, &result))
        return result;
    return DefWindowProc(window, message, wParam, lParam);
}

DGL_Mesh* MakeQuad(DGL_Color color)
{
    const DGL_Vec2 bl{-0.5f, -0.5f}, br{0.5f, -0.5f}, tr{0.5f, 0.5f}, tl{-0.5f, 0.5f};
    const DGL_Vec2 uvBl{0, 1}, uvBr{1, 1}, uvTr{1, 0}, uvTl{0, 0};
    DGL_Graphics_StartMesh();
    DGL_Graphics_AddTriangle(&bl, &color, &uvBl, &tr, &color, &uvTr, &br, &color, &uvBr);
    DGL_Graphics_AddTriangle(&bl, &color, &uvBl, &tl, &color, &uvTl, &tr, &color, &uvTr);
    return DGL_Graphics_EndMesh();
}

DGL_Mesh* MakeTriangle()
{
    const DGL_Vec2 top{0.0f, 0.5f}, left{-0.5f, -0.5f}, right{0.5f, -0.5f};
    const DGL_Color red{1, 0, 0, 1}, green{0, 1, 0, 1}, blue{0, 0, 1, 1};
    const DGL_Vec2 uv{0, 0};
    DGL_Graphics_StartMesh();
    DGL_Graphics_AddTriangle(&top, &red, &uv, &left, &green, &uv, &right, &blue, &uv);
    return DGL_Graphics_EndMesh();
}

DGL_Texture* MakeChecker()
{
    constexpr int kSize = 8;
    unsigned char pixels[kSize * kSize * 4];
    for (int y = 0; y < kSize; ++y)
        for (int x = 0; x < kSize; ++x)
        {
            const bool light = (x + y) % 2 == 0;
            unsigned char* p = &pixels[(y * kSize + x) * 4];
            p[0] = light ? 250 : 40;
            p[1] = light ? 200 : 40;
            p[2] = light ? 60 : 90;
            p[3] = 255;
        }
    return DGL_Graphics_LoadTextureFromMemory(pixels, kSize, kSize);
}

} // namespace

int APIENTRY wWinMain(_In_ HINSTANCE instance, _In_opt_ HINSTANCE, _In_ LPWSTR, _In_ int show)
{
    DGL_SysInitInfo info{};
    info.mAppInstance = instance;
    info.mShow = show;
    info.mWindowWidth = 800;
    info.mWindowHeight = 600;
    info.mMaxFrameRate = 60;
    info.mClassStyle = CS_HREDRAW | CS_VREDRAW;
    info.mWindowStyle = WS_OVERLAPPEDWINDOW;
    info.mWindowTitle = "DGL compat demo";
    info.pWindowsCallback = WindowProc;
    if (!DGL_System_Init(&info))
    {
        std::fprintf(stderr, "init failed: %s\n", DGL_System_GetLastError());
        return 1;
    }

    const char* framesEnv = std::getenv("DGL_DEMO_FRAMES");
    const unsigned maxFrames = framesEnv ? static_cast<unsigned>(std::atoi(framesEnv)) : 0;

    DGL_Mesh* white = MakeQuad({1, 1, 1, 1});
    DGL_Mesh* black = MakeQuad({0, 0, 0, 1});
    DGL_Mesh* triangle = MakeTriangle();
    DGL_Texture* checker = MakeChecker();

    const DGL_Color background{0.15f, 0.16f, 0.2f, 1.0f};
    const DGL_Color noTint{0, 0, 0, 0};
    const DGL_Color halfRed{1, 0, 0, 0.5f};
    DGL_Graphics_SetBackgroundColor(&background);
    DGL_Graphics_SetBlendMode(DGL_BM_BLEND);
    DGL_Graphics_SetTextureSamplerData(DGL_TSM_POINT, DGL_AM_WRAP);

    DGL_Vec2 camera{0, 0};
    float zoom = 1.0f;
    float angle = 0.0f;

    while (DGL_System_DoesWindowExist())
    {
        DGL_System_FrameControl();
        DGL_System_Update();
        if (DGL_Input_KeyTriggered(VK_ESCAPE) ||
            (maxFrames && DGL_System_GetFrameCount() >= maxFrames))
            break;

        const float dt = static_cast<float>(DGL_System_GetDeltaTime());
        angle += dt;
        if (DGL_Input_KeyDown(VK_LEFT)) camera.x -= 300 * dt;
        if (DGL_Input_KeyDown(VK_RIGHT)) camera.x += 300 * dt;
        if (DGL_Input_KeyDown(VK_UP)) camera.y += 300 * dt;
        if (DGL_Input_KeyDown(VK_DOWN)) camera.y -= 300 * dt;
        if (DGL_Input_KeyDown('Z')) zoom *= 1.0f - dt;
        if (DGL_Input_KeyDown('X')) zoom *= 1.0f + dt;
        DGL_Camera_SetPosition(&camera);
        DGL_Camera_SetZoom(zoom);

        DGL_Graphics_StartDrawing();

        // Vertex colors, rotating.
        const DGL_Vec2 triPos{-220, 0}, triScale{200, 200};
        DGL_Graphics_SetShaderMode(DGL_PSM_COLOR, DGL_VSM_DEFAULT);
        DGL_Graphics_SetCB_Alpha(1.0f);
        DGL_Graphics_SetCB_TintColor(&noTint);
        DGL_Graphics_SetCB_TransformData(&triPos, &triScale, angle);
        DGL_Graphics_DrawMesh(triangle, DGL_DM_TRIANGLELIST);

        // Texture (black vertices are ignored), drawn with a matrix and a half-red tint.
        DGL_Mat4 m{};
        m.m[0][0] = 200; m.m[1][1] = 200; m.m[2][2] = 1; m.m[3][3] = 1;
        m.m[0][3] = 220; m.m[1][3] = 0;
        DGL_Graphics_SetShaderMode(DGL_PSM_TEXTURE, DGL_VSM_DEFAULT);
        DGL_Graphics_SetTexture(checker);
        const DGL_Vec2 noOffset{0, 0};
        DGL_Graphics_SetCB_TextureOffset(&noOffset);
        DGL_Graphics_SetCB_TintColor(&halfRed);
        DGL_Graphics_SetCB_TransformMatrix(&m);
        DGL_Graphics_DrawMesh(black, DGL_DM_TRIANGLELIST);

        // A translucent white square under the mouse, placed in world coordinates.
        const DGL_Vec2 mouse = DGL_Input_GetMousePosition();
        const DGL_Vec2 world = DGL_Camera_ScreenCoordToWorld(&mouse);
        const DGL_Vec2 cursorScale{40, 40};
        DGL_Graphics_SetShaderMode(DGL_PSM_COLOR, DGL_VSM_DEFAULT);
        DGL_Graphics_SetCB_TintColor(&noTint);
        DGL_Graphics_SetCB_Alpha(DGL_Input_KeyDown(VK_LBUTTON) ? 1.0f : 0.5f);
        DGL_Graphics_SetCB_TransformData(&world, &cursorScale, 0.0f);
        DGL_Graphics_DrawMesh(white, DGL_DM_TRIANGLELIST);

        DGL_Graphics_FinishDrawing();
    }

    DGL_Graphics_FreeTexture(&checker);
    DGL_Graphics_FreeMesh(&triangle);
    DGL_Graphics_FreeMesh(&black);
    DGL_Graphics_FreeMesh(&white);
    return DGL_System_Exit();
}
