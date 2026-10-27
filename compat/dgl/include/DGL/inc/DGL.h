// DGL.h — DGL 1.4 API, implemented on GLFW + OpenGL 3.3 so DGL programs run on macOS and Linux.
//
// This is an independent implementation of the public DGL interface (as documented at
// https://github.com/DigiPen-Faculty/DigiPen-Graphics-Library/wiki). It contains no DigiPen code.
// Declarations mirror DGL 1.4.0 so existing programs compile unchanged; behavior that the
// documentation leaves open is noted in compat/README.md.

#pragma once

#include <Windows.h>

typedef struct DGL_Vec2
{
    float x;
    float y;
} DGL_Vec2;

// Row-major: m[row][column]. Column-vector convention, so translation lives in m[0][3], m[1][3].
typedef struct DGL_Mat4
{
    float m[4][4];
} DGL_Mat4;

typedef struct DGL_Color
{
    float r;
    float g;
    float b;
    float a;
} DGL_Color;

typedef struct DGL_SysInitInfo
{
    HINSTANCE mAppInstance;
    int mShow;
    unsigned int mWindowWidth;
    unsigned int mWindowHeight;
    unsigned int mMaxFrameRate;
    unsigned int mClassStyle;
    unsigned int mWindowStyle;
    const char* mWindowTitle;
    BOOL mCreateConsole;
    int mWindowIcon;
    WNDPROC pWindowsCallback;
} DGL_SysInitInfo;

typedef struct DGL_Texture DGL_Texture;
typedef struct DGL_Mesh DGL_Mesh;
typedef struct DGL_PixelShader DGL_PixelShader;
typedef struct DGL_VertexShader DGL_VertexShader;

typedef enum
{
    DGL_DM_TRIANGLELIST,
    DGL_DM_TRIANGLESTRIP,
    DGL_DM_LINELIST,
    DGL_DM_LINESTRIP,
    DGL_DM_POINTLIST,
} DGL_DrawMode;

typedef enum
{
    DGL_BM_NONE,
    DGL_BM_BLEND,
    DGL_BM_ADD,
    DGL_BM_MULTIPLY,
} DGL_BlendMode;

typedef enum
{
    DGL_TSM_LINEAR,
    DGL_TSM_POINT,
} DGL_TextureSampleMode;

typedef enum
{
    DGL_AM_WRAP,
    DGL_AM_MIRROR,
    DGL_AM_CLAMP,
    DGL_AM_MIRROR_ONCE,
} DGL_TextureAddressMode;

typedef enum
{
    DGL_PSM_COLOR,
    DGL_PSM_TEXTURE,
    DGL_PSM_CUSTOM,
} DGL_PixelShaderMode;

typedef enum
{
    DGL_VSM_DEFAULT,
    DGL_VSM_CUSTOM,
} DGL_VertexShaderMode;

#ifdef __cplusplus
extern "C"
{
#endif

// System
HWND DGL_System_Init(const DGL_SysInitInfo* sysInitInfo);
void DGL_System_Reset(void);
void DGL_System_FrameControl(void);
void DGL_System_Update(void);
BOOL DGL_System_HandleWindowsMessage(UINT message, WPARAM wParam, LPARAM lParam, int* result);
int DGL_System_Exit(void);
BOOL DGL_System_DoesWindowExist(void);
HWND DGL_System_GetWindowHandle(void);
double DGL_System_GetDeltaTime(void);
unsigned DGL_System_GetFrameCount(void);
double DGL_System_GetTime(void);
const char* DGL_System_GetLastError(void);

// Camera
DGL_Vec2 DGL_Camera_ScreenCoordToWorld(const DGL_Vec2* position);
DGL_Vec2 DGL_Camera_GetPosition(void);
void DGL_Camera_SetPosition(const DGL_Vec2* position);
float DGL_Camera_GetZoom(void);
void DGL_Camera_SetZoom(float zoom);
float DGL_Camera_GetRotation(void);
void DGL_Camera_SetRotation(float radians);

// Graphics: settings
void DGL_Graphics_SetBackgroundColor(const DGL_Color* color);
void DGL_Graphics_SetTextureSamplerData(DGL_TextureSampleMode sampleMode,
                                        DGL_TextureAddressMode addressMode);
void DGL_Graphics_SetBlendMode(DGL_BlendMode mode);
void DGL_Graphics_SetShaderMode(DGL_PixelShaderMode pixelMode, DGL_VertexShaderMode vertexMode);
void DGL_Graphics_SetCustomPixelShader(const DGL_PixelShader* shader);
void DGL_Graphics_SetCustomVertexShader(const DGL_VertexShader* shader);
void DGL_Graphics_SetTexture(const DGL_Texture* texture);

// Graphics: shaders (HLSL is not supported on OpenGL; loading reports an error)
const DGL_PixelShader* DGL_Graphics_LoadPixelShader(const char* filename);
const DGL_VertexShader* DGL_Graphics_LoadVertexShader(const char* filename);
void DGL_Graphics_FreePixelShader(const DGL_PixelShader** shader);
void DGL_Graphics_FreeVertexShader(const DGL_VertexShader** shader);

// Graphics: textures
DGL_Texture* DGL_Graphics_LoadTexture(const char* fileName);
DGL_Texture* DGL_Graphics_LoadTextureFromMemory(const unsigned char* data, int width, int height);
DGL_Texture* DGL_Graphics_CreateRenderTexture(int width, int height);
void DGL_Graphics_ClearRenderTexture(const DGL_Texture* renderTexture, const DGL_Color* color);
void DGL_Graphics_FreeTexture(DGL_Texture** texture);
DGL_Vec2 DGL_Graphics_GetTextureSize(DGL_Texture* texture);

// Graphics: meshes
void DGL_Graphics_StartMesh(void);
DGL_Mesh* DGL_Graphics_EndMesh(void);
DGL_Mesh* DGL_Graphics_EndMeshIndexed(unsigned* indices, unsigned indexCount);
void DGL_Graphics_AddVertex(const DGL_Vec2* position, const DGL_Color* color,
                            const DGL_Vec2* textureOffset);
void DGL_Graphics_AddTriangle(const DGL_Vec2* position1, const DGL_Color* color1,
                              const DGL_Vec2* textureOffset1, const DGL_Vec2* position2,
                              const DGL_Color* color2, const DGL_Vec2* textureOffset2,
                              const DGL_Vec2* position3, const DGL_Color* color3,
                              const DGL_Vec2* textureOffset3);
void DGL_Graphics_FreeMesh(DGL_Mesh** mesh);

// Graphics: drawing
void DGL_Graphics_StartDrawing(void);
void DGL_Graphics_FinishDrawing(void);
void DGL_Graphics_DrawMesh(const DGL_Mesh* mesh, DGL_DrawMode mode);
void DGL_Graphics_DrawMeshToTexture(const DGL_Mesh* mesh, DGL_DrawMode mode,
                                    const DGL_Texture* renderTexture);

// Graphics: per-draw constants
void DGL_Graphics_SetCB_TransformData(const DGL_Vec2* position, const DGL_Vec2* scale,
                                      float rotationRadians);
void DGL_Graphics_SetCB_TransformMatrix(const DGL_Mat4* transformationMatrix);
void DGL_Graphics_SetCB_TextureOffset(const DGL_Vec2* textureOffset);
void DGL_Graphics_SetCB_Alpha(float alpha);
void DGL_Graphics_SetCB_TintColor(const DGL_Color* color);
void DGL_Graphics_SetCB_ShaderData(float data);

// Input (keys are Win32 virtual-key codes, including VK_LBUTTON / VK_RBUTTON / VK_MBUTTON)
int DGL_Input_GetMouseScrollDelta(void);
DGL_Vec2 DGL_Input_GetMousePosition(void);
DGL_Vec2 DGL_Input_GetMousePositionDelta(void);
BOOL DGL_Input_KeyDown(unsigned char key);
BOOL DGL_Input_KeyTriggered(unsigned char key);
BOOL DGL_Input_KeyReleased(unsigned char key);
unsigned char DGL_Input_LastKeyTriggered(void);
void DGL_Input_ShowCursor(BOOL show);

// Window
BOOL DGL_Window_HasFocus(void);
DGL_Vec2 DGL_Window_GetMonitorSize(void);
DGL_Vec2 DGL_Window_GetSize(void);
void DGL_Window_SetSizeAndPosition(int windowWidth, int windowHeight, int windowLeftPos,
                                   int windowTopPos);
void DGL_Window_SetSize(int windowWidth, int windowHeight);
void DGL_Window_SetStyle(unsigned style);

#ifdef __cplusplus
}
#endif
