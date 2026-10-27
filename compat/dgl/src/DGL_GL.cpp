// DGL 1.4 API on GLFW + OpenGL 3.3 core. See DGL.h and compat/README.md.

#include <DGL.h>

#define GLAD_GL_IMPLEMENTATION
#include <glad/gl.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#define STBI_ONLY_BMP
#define STBI_ONLY_TGA
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

struct DGL_Texture
{
    GLuint id = 0;
    int width = 0;
    int height = 0;
    GLuint framebuffer = 0; // non-zero for render textures
};

struct DGL_Mesh
{
    GLuint vao = 0;
    GLuint vbo = 0;
    GLuint ebo = 0;
    GLsizei count = 0;
    bool indexed = false;
};

namespace
{

struct Vertex
{
    float x, y;
    float r, g, b, a;
    float u, v;
};

constexpr const char* kVertexShader = R"(#version 330 core
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec4 aColor;
layout(location = 2) in vec2 aUV;
uniform mat4 uModel;
uniform mat4 uViewProj;
uniform vec2 uTexOffset;
out vec4 vColor;
out vec2 vUV;
void main()
{
    gl_Position = uViewProj * uModel * vec4(aPos, 0.0, 1.0);
    vColor = aColor;
    vUV = aUV + uTexOffset;
}
)";

// Tint is blended by its own alpha, so the common "no tint" value {0,0,0,0} leaves colors alone.
// The texture shader ignores vertex color: DGL sprite meshes are built with black vertices.
constexpr const char* kFragmentShader = R"(#version 330 core
in vec4 vColor;
in vec2 vUV;
uniform int uTextured;
uniform sampler2D uTexture;
uniform vec4 uTint;
uniform float uAlpha;
out vec4 fragColor;
void main()
{
    vec4 color = (uTextured != 0) ? texture(uTexture, vUV) : vColor;
    color.rgb = mix(color.rgb, uTint.rgb, uTint.a);
    color.a *= uAlpha;
    fragColor = color;
}
)";

struct Uniforms
{
    GLint model = -1, viewProj = -1, texOffset = -1, textured = -1, texture = -1, tint = -1,
          alpha = -1;
};

struct State
{
    GLFWwindow* window = nullptr;
    std::string title;
    std::string lastError;

    GLuint program = 0;
    Uniforms uniforms;
    GLuint sampler = 0;
    GLuint whiteTexture = 0; // bound when drawing untextured, so the sampler is never empty

    // Frame timing
    double frameTarget = 1.0 / 60.0;
    double lastFrameTime = 0.0;
    double deltaTime = 0.0;
    unsigned frameCount = 0;

    // Camera
    DGL_Vec2 cameraPosition{0.0f, 0.0f};
    float cameraZoom = 1.0f;
    float cameraRotation = 0.0f;

    // Per-draw constants
    DGL_Color background{0.0f, 0.0f, 0.0f, 1.0f};
    std::array<float, 16> model{}; // column-major for glUniformMatrix4fv
    DGL_Vec2 texOffset{0.0f, 0.0f};
    DGL_Color tint{0.0f, 0.0f, 0.0f, 0.0f};
    float alpha = 1.0f;
    DGL_PixelShaderMode pixelMode = DGL_PSM_COLOR;
    const DGL_Texture* texture = nullptr;

    // Mesh under construction
    std::vector<Vertex> building;

    // Input, indexed by virtual-key code
    std::array<bool, 256> keyDown{};
    std::array<bool, 256> keyWasDown{};
    unsigned char lastKeyTriggered = 0;
    DGL_Vec2 mouse{0.0f, 0.0f};
    DGL_Vec2 previousMouse{0.0f, 0.0f};
    double scrollAccumulator = 0.0;
    int scrollDelta = 0;
    WNDPROC windowProc = nullptr; // gets WM_CHAR for typed characters, as on Windows
};

State g;

// Testing hooks, read once at startup:
//   DGL_COMPAT_CAPTURE=out.png@120[,out2.png@240...]  save the frame drawn on frame N as PNG
//   DGL_COMPAT_EXIT_AFTER=N                           close the window after N frames
//   DGL_COMPAT_HOLD=0x27@30-90[,0x26@50-52...]        hold a virtual key during frames [from, to]
struct Capture
{
    std::string path;
    unsigned frame = 0;
};
std::vector<Capture> g_captures;
unsigned g_exitAfter = 0;

struct Hold
{
    int key = 0;
    unsigned from = 0;
    unsigned to = 0;
};
std::vector<Hold> g_holds;

// Splits "a,b,c" and hands each item to `parse`.
template <typename Parse> void ForEachItem(const char* spec, Parse parse)
{
    const std::string list = spec;
    size_t start = 0;
    while (start < list.size())
    {
        const size_t end = std::min(list.find(',', start), list.size());
        parse(list.substr(start, end - start));
        start = end + 1;
    }
}

void ReadTestHooks()
{
    if (const char* exitAfter = std::getenv("DGL_COMPAT_EXIT_AFTER"))
        g_exitAfter = static_cast<unsigned>(std::strtoul(exitAfter, nullptr, 10));
    if (const char* spec = std::getenv("DGL_COMPAT_CAPTURE"))
        ForEachItem(spec, [](const std::string& item) {
            const size_t at = item.rfind('@');
            if (at != std::string::npos)
                g_captures.push_back(
                    {item.substr(0, at),
                     static_cast<unsigned>(std::strtoul(item.c_str() + at + 1, nullptr, 10))});
        });
    if (const char* spec = std::getenv("DGL_COMPAT_HOLD"))
        ForEachItem(spec, [](const std::string& item) {
            const size_t at = item.find('@');
            const size_t dash = item.find('-', at);
            if (at == std::string::npos || dash == std::string::npos)
                return;
            Hold hold;
            hold.key = static_cast<int>(std::strtol(item.c_str(), nullptr, 0)); // 0x27 or 39
            hold.from = static_cast<unsigned>(std::strtoul(item.c_str() + at + 1, nullptr, 10));
            hold.to = static_cast<unsigned>(std::strtoul(item.c_str() + dash + 1, nullptr, 10));
            g_holds.push_back(hold);
        });
}

bool HeldByTestHook(int vk)
{
    for (const Hold& hold : g_holds)
        if (hold.key == vk && g.frameCount >= hold.from && g.frameCount <= hold.to)
            return true;
    return false;
}

void SetError(std::string message)
{
    std::fprintf(stderr, "[DGL compat] %s\n", message.c_str());
    g.lastError = std::move(message);
}

void SetIdentity(std::array<float, 16>& m)
{
    m = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
}

// Virtual-key code -> GLFW key(s). Letters and digits share their ASCII codes in both.
int MapVirtualKey(int vk, int& second)
{
    second = GLFW_KEY_UNKNOWN;
    if ((vk >= 'A' && vk <= 'Z') || (vk >= '0' && vk <= '9'))
        return vk;
    if (vk >= VK_F1 && vk <= VK_F12)
        return GLFW_KEY_F1 + (vk - VK_F1);
    if (vk >= VK_NUMPAD0 && vk <= VK_NUMPAD9)
        return GLFW_KEY_KP_0 + (vk - VK_NUMPAD0);

    switch (vk)
    {
    case VK_BACK: return GLFW_KEY_BACKSPACE;
    case VK_TAB: return GLFW_KEY_TAB;
    case VK_RETURN: second = GLFW_KEY_KP_ENTER; return GLFW_KEY_ENTER;
    case VK_SHIFT: second = GLFW_KEY_RIGHT_SHIFT; return GLFW_KEY_LEFT_SHIFT;
    case VK_CONTROL: second = GLFW_KEY_RIGHT_CONTROL; return GLFW_KEY_LEFT_CONTROL;
    case VK_MENU: second = GLFW_KEY_RIGHT_ALT; return GLFW_KEY_LEFT_ALT;
    case VK_PAUSE: return GLFW_KEY_PAUSE;
    case VK_CAPITAL: return GLFW_KEY_CAPS_LOCK;
    case VK_ESCAPE: return GLFW_KEY_ESCAPE;
    case VK_SPACE: return GLFW_KEY_SPACE;
    case VK_PRIOR: return GLFW_KEY_PAGE_UP;
    case VK_NEXT: return GLFW_KEY_PAGE_DOWN;
    case VK_END: return GLFW_KEY_END;
    case VK_HOME: return GLFW_KEY_HOME;
    case VK_LEFT: return GLFW_KEY_LEFT;
    case VK_UP: return GLFW_KEY_UP;
    case VK_RIGHT: return GLFW_KEY_RIGHT;
    case VK_DOWN: return GLFW_KEY_DOWN;
    case VK_INSERT: return GLFW_KEY_INSERT;
    case VK_DELETE: return GLFW_KEY_DELETE;
    case VK_MULTIPLY: return GLFW_KEY_KP_MULTIPLY;
    case VK_ADD: return GLFW_KEY_KP_ADD;
    case VK_SUBTRACT: return GLFW_KEY_KP_SUBTRACT;
    case VK_DECIMAL: return GLFW_KEY_KP_DECIMAL;
    case VK_DIVIDE: return GLFW_KEY_KP_DIVIDE;
    case VK_NUMLOCK: return GLFW_KEY_NUM_LOCK;
    case VK_SCROLL: return GLFW_KEY_SCROLL_LOCK;
    case VK_LSHIFT: return GLFW_KEY_LEFT_SHIFT;
    case VK_RSHIFT: return GLFW_KEY_RIGHT_SHIFT;
    case VK_LCONTROL: return GLFW_KEY_LEFT_CONTROL;
    case VK_RCONTROL: return GLFW_KEY_RIGHT_CONTROL;
    case VK_LMENU: return GLFW_KEY_LEFT_ALT;
    case VK_RMENU: return GLFW_KEY_RIGHT_ALT;
    case VK_OEM_1: return GLFW_KEY_SEMICOLON;
    case VK_OEM_PLUS: return GLFW_KEY_EQUAL;
    case VK_OEM_COMMA: return GLFW_KEY_COMMA;
    case VK_OEM_MINUS: return GLFW_KEY_MINUS;
    case VK_OEM_PERIOD: return GLFW_KEY_PERIOD;
    case VK_OEM_2: return GLFW_KEY_SLASH;
    case VK_OEM_3: return GLFW_KEY_GRAVE_ACCENT;
    case VK_OEM_4: return GLFW_KEY_LEFT_BRACKET;
    case VK_OEM_5: return GLFW_KEY_BACKSLASH;
    case VK_OEM_6: return GLFW_KEY_RIGHT_BRACKET;
    case VK_OEM_7: return GLFW_KEY_APOSTROPHE;
    default: return GLFW_KEY_UNKNOWN;
    }
}

bool QueryVirtualKey(int vk)
{
    switch (vk)
    {
    case VK_LBUTTON: return glfwGetMouseButton(g.window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
    case VK_RBUTTON: return glfwGetMouseButton(g.window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
    case VK_MBUTTON: return glfwGetMouseButton(g.window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS;
    default: break;
    }
    int second = GLFW_KEY_UNKNOWN;
    const int key = MapVirtualKey(vk, second);
    if (key == GLFW_KEY_UNKNOWN)
        return false;
    return glfwGetKey(g.window, key) == GLFW_PRESS ||
           (second != GLFW_KEY_UNKNOWN && glfwGetKey(g.window, second) == GLFW_PRESS);
}

void WindowSize(float& w, float& h)
{
    int iw = 1, ih = 1;
    glfwGetWindowSize(g.window, &iw, &ih);
    w = static_cast<float>(iw > 0 ? iw : 1);
    h = static_cast<float>(ih > 0 ? ih : 1);
}

// World -> clip for a target of the given size: origin at the center, y up, 1 unit = 1 pixel at
// zoom 1. Smaller zoom shows less of the world (objects look larger), matching DGL.
std::array<float, 16> ViewProjection(float targetWidth, float targetHeight)
{
    const float c = std::cos(-g.cameraRotation);
    const float s = std::sin(-g.cameraRotation);
    const float sx = 2.0f / (targetWidth * g.cameraZoom);
    const float sy = 2.0f / (targetHeight * g.cameraZoom);
    const float tx = -(c * g.cameraPosition.x - s * g.cameraPosition.y);
    const float ty = -(s * g.cameraPosition.x + c * g.cameraPosition.y);
    // Column-major.
    return {sx * c, sy * s, 0, 0, -sx * s, sy * c, 0, 0, 0, 0, 1, 0, sx * tx, sy * ty, 0, 1};
}

GLuint CompileShader(GLenum type, const char* source)
{
    const GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);
    GLint ok = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (!ok)
    {
        char log[1024];
        glGetShaderInfoLog(shader, sizeof(log), nullptr, log);
        SetError(std::string("shader compile error: ") + log);
        glDeleteShader(shader);
        return 0;
    }
    return shader;
}

bool CreateProgram()
{
    const GLuint vs = CompileShader(GL_VERTEX_SHADER, kVertexShader);
    const GLuint fs = CompileShader(GL_FRAGMENT_SHADER, kFragmentShader);
    if (!vs || !fs)
        return false;
    g.program = glCreateProgram();
    glAttachShader(g.program, vs);
    glAttachShader(g.program, fs);
    glLinkProgram(g.program);
    glDeleteShader(vs);
    glDeleteShader(fs);
    GLint ok = GL_FALSE;
    glGetProgramiv(g.program, GL_LINK_STATUS, &ok);
    if (!ok)
    {
        SetError("shader link error");
        return false;
    }
    Uniforms& u = g.uniforms;
    u.model = glGetUniformLocation(g.program, "uModel");
    u.viewProj = glGetUniformLocation(g.program, "uViewProj");
    u.texOffset = glGetUniformLocation(g.program, "uTexOffset");
    u.textured = glGetUniformLocation(g.program, "uTextured");
    u.texture = glGetUniformLocation(g.program, "uTexture");
    u.tint = glGetUniformLocation(g.program, "uTint");
    u.alpha = glGetUniformLocation(g.program, "uAlpha");
    return true;
}

DGL_Mesh* UploadMesh(const std::vector<Vertex>& vertices, const unsigned* indices,
                     unsigned indexCount)
{
    if (vertices.empty())
    {
        SetError("EndMesh called with no vertices");
        return nullptr;
    }
    auto* mesh = new DGL_Mesh;
    glGenVertexArrays(1, &mesh->vao);
    glGenBuffers(1, &mesh->vbo);
    glBindVertexArray(mesh->vao);
    glBindBuffer(GL_ARRAY_BUFFER, mesh->vbo);
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(vertices.size() * sizeof(Vertex)),
                 vertices.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                          reinterpret_cast<void*>(offsetof(Vertex, x)));
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                          reinterpret_cast<void*>(offsetof(Vertex, r)));
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                          reinterpret_cast<void*>(offsetof(Vertex, u)));
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glEnableVertexAttribArray(2);

    if (indices && indexCount > 0)
    {
        glGenBuffers(1, &mesh->ebo);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh->ebo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, static_cast<GLsizeiptr>(indexCount * sizeof(unsigned)),
                     indices, GL_STATIC_DRAW);
        mesh->indexed = true;
        mesh->count = static_cast<GLsizei>(indexCount);
    }
    else
    {
        mesh->count = static_cast<GLsizei>(vertices.size());
    }
    glBindVertexArray(0);
    return mesh;
}

DGL_Texture* CreateTexture(const unsigned char* rgba, int width, int height)
{
    auto* texture = new DGL_Texture;
    texture->width = width;
    texture->height = height;
    glGenTextures(1, &texture->id);
    glBindTexture(GL_TEXTURE_2D, texture->id);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, rgba);
    // Filtering and wrapping come from the shared sampler object.
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
    return texture;
}

GLenum ToGLMode(DGL_DrawMode mode)
{
    switch (mode)
    {
    case DGL_DM_TRIANGLELIST: return GL_TRIANGLES;
    case DGL_DM_TRIANGLESTRIP: return GL_TRIANGLE_STRIP;
    case DGL_DM_LINELIST: return GL_LINES;
    case DGL_DM_LINESTRIP: return GL_LINE_STRIP;
    case DGL_DM_POINTLIST: return GL_POINTS;
    }
    return GL_TRIANGLES;
}

void Draw(const DGL_Mesh* mesh, DGL_DrawMode mode, float targetWidth, float targetHeight)
{
    if (!mesh)
        return;
    const Uniforms& u = g.uniforms;
    const bool textured = g.pixelMode == DGL_PSM_TEXTURE && g.texture;

    glUseProgram(g.program);
    const auto viewProj = ViewProjection(targetWidth, targetHeight);
    glUniformMatrix4fv(u.viewProj, 1, GL_FALSE, viewProj.data());
    glUniformMatrix4fv(u.model, 1, GL_FALSE, g.model.data());
    glUniform2f(u.texOffset, g.texOffset.x, g.texOffset.y);
    glUniform4f(u.tint, g.tint.r, g.tint.g, g.tint.b, g.tint.a);
    glUniform1f(u.alpha, g.alpha);
    glUniform1i(u.textured, textured ? 1 : 0);
    glUniform1i(u.texture, 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, textured ? g.texture->id : g.whiteTexture);
    glBindSampler(0, g.sampler);

    glBindVertexArray(mesh->vao);
    if (mesh->indexed)
        glDrawElements(ToGLMode(mode), mesh->count, GL_UNSIGNED_INT, nullptr);
    else
        glDrawArrays(ToGLMode(mode), 0, mesh->count);
    glBindVertexArray(0);
}

void OnScroll(GLFWwindow*, double, double yOffset) { g.scrollAccumulator += yOffset; }

// Windows sends typed characters to the window procedure as WM_CHAR (UTF-16 code units); so does this.
void OnChar(GLFWwindow*, unsigned codepoint)
{
    if (g.windowProc && codepoint < 0x10000)
        g.windowProc(nullptr, WM_CHAR, static_cast<WPARAM>(codepoint), 0);
}

} // namespace

extern "C"
{

// --- System ------------------------------------------------------------------

HWND DGL_System_Init(const DGL_SysInitInfo* info)
{
    if (!info)
    {
        SetError("DGL_System_Init: info is null");
        return nullptr;
    }
    ReadTestHooks();
    if (!glfwInit())
    {
        SetError("glfwInit failed");
        return nullptr;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
    glfwWindowHint(GLFW_RESIZABLE, (info->mWindowStyle & WS_THICKFRAME) ? GLFW_TRUE : GLFW_FALSE);
    glfwWindowHint(GLFW_VISIBLE, info->mShow != SW_HIDE ? GLFW_TRUE : GLFW_FALSE);

    g.title = info->mWindowTitle ? info->mWindowTitle : "DGL";
    const int width = info->mWindowWidth ? static_cast<int>(info->mWindowWidth) : 1024;
    const int height = info->mWindowHeight ? static_cast<int>(info->mWindowHeight) : 768;
    g.window = glfwCreateWindow(width, height, g.title.c_str(), nullptr, nullptr);
    if (!g.window)
    {
        SetError("glfwCreateWindow failed");
        glfwTerminate();
        return nullptr;
    }
    glfwMakeContextCurrent(g.window);
    glfwSwapInterval(1);
    glfwSetScrollCallback(g.window, OnScroll);
    glfwSetCharCallback(g.window, OnChar);
    g.windowProc = info->pWindowsCallback;

    if (!gladLoadGL(glfwGetProcAddress) || !CreateProgram())
    {
        SetError("could not initialize OpenGL 3.3");
        return nullptr;
    }

    glGenSamplers(1, &g.sampler);
    const unsigned char white[4] = {255, 255, 255, 255};
    DGL_Texture* whiteTexture = CreateTexture(white, 1, 1);
    g.whiteTexture = whiteTexture->id;
    delete whiteTexture;
    DGL_Graphics_SetTextureSamplerData(DGL_TSM_LINEAR, DGL_AM_WRAP);
    DGL_Graphics_SetBlendMode(DGL_BM_NONE);
    SetIdentity(g.model);

    g.frameTarget = info->mMaxFrameRate ? 1.0 / info->mMaxFrameRate : 0.0;
    g.lastFrameTime = glfwGetTime();
    g.frameCount = 0;
    return reinterpret_cast<HWND>(g.window);
}

void DGL_System_Reset(void)
{
    g.frameCount = 0;
    g.lastFrameTime = glfwGetTime();
    g.cameraPosition = {0.0f, 0.0f};
    g.cameraZoom = 1.0f;
    g.cameraRotation = 0.0f;
}

void DGL_System_FrameControl(void)
{
    if (g.frameTarget > 0.0)
    {
        const double wakeAt = g.lastFrameTime + g.frameTarget;
        double now = glfwGetTime();
        if (wakeAt - now > 0.002)
            std::this_thread::sleep_for(std::chrono::duration<double>(wakeAt - now - 0.001));
        while (glfwGetTime() < wakeAt)
        {
        }
    }
    const double now = glfwGetTime();
    g.deltaTime = now - g.lastFrameTime;
    g.lastFrameTime = now;
    ++g.frameCount;
}

void DGL_System_Update(void)
{
    if (!g.window)
        return;
    g.scrollAccumulator = 0.0;
    glfwPollEvents();

    g.keyWasDown = g.keyDown;
    g.lastKeyTriggered = 0;
    for (int vk = 1; vk < 256; ++vk)
    {
        g.keyDown[vk] = QueryVirtualKey(vk) || HeldByTestHook(vk);
        if (g.keyDown[vk] && !g.keyWasDown[vk] && vk > VK_MBUTTON)
            g.lastKeyTriggered = static_cast<unsigned char>(vk);
    }

    double x = 0.0, y = 0.0;
    glfwGetCursorPos(g.window, &x, &y);
    g.previousMouse = g.mouse;
    g.mouse = {static_cast<float>(x), static_cast<float>(y)};
    g.scrollDelta = static_cast<int>(std::lround(g.scrollAccumulator));
}

BOOL DGL_System_HandleWindowsMessage(UINT, WPARAM, LPARAM, int* result)
{
    if (result)
        *result = 0;
    return FALSE; // GLFW handles window events itself
}

int DGL_System_Exit(void)
{
    if (!g.window)
        return 0;
    glDeleteSamplers(1, &g.sampler);
    glDeleteTextures(1, &g.whiteTexture);
    glDeleteProgram(g.program);
    glfwDestroyWindow(g.window);
    glfwTerminate();
    g.window = nullptr;
    return 0;
}

BOOL DGL_System_DoesWindowExist(void)
{
    return g.window && !glfwWindowShouldClose(g.window) ? TRUE : FALSE;
}

HWND DGL_System_GetWindowHandle(void) { return reinterpret_cast<HWND>(g.window); }
double DGL_System_GetDeltaTime(void) { return g.deltaTime; }
unsigned DGL_System_GetFrameCount(void) { return g.frameCount; }
double DGL_System_GetTime(void) { return glfwGetTime(); }
const char* DGL_System_GetLastError(void) { return g.lastError.c_str(); }

// --- Camera ------------------------------------------------------------------

DGL_Vec2 DGL_Camera_ScreenCoordToWorld(const DGL_Vec2* position)
{
    float w = 1.0f, h = 1.0f;
    WindowSize(w, h);
    const float vx = (position->x - w * 0.5f) * g.cameraZoom;
    const float vy = (h * 0.5f - position->y) * g.cameraZoom;
    const float c = std::cos(g.cameraRotation);
    const float s = std::sin(g.cameraRotation);
    return {c * vx - s * vy + g.cameraPosition.x, s * vx + c * vy + g.cameraPosition.y};
}

DGL_Vec2 DGL_Camera_GetPosition(void) { return g.cameraPosition; }
void DGL_Camera_SetPosition(const DGL_Vec2* position) { g.cameraPosition = *position; }
float DGL_Camera_GetZoom(void) { return g.cameraZoom; }
void DGL_Camera_SetZoom(float zoom) { g.cameraZoom = zoom > 1e-6f ? zoom : 1e-6f; }
float DGL_Camera_GetRotation(void) { return g.cameraRotation; }
void DGL_Camera_SetRotation(float radians) { g.cameraRotation = radians; }

// --- Graphics: settings ------------------------------------------------------

void DGL_Graphics_SetBackgroundColor(const DGL_Color* color)
{
    g.background = *color;
    g.background.a = 1.0f; // DGL ignores background alpha
}

void DGL_Graphics_SetTextureSamplerData(DGL_TextureSampleMode sampleMode,
                                        DGL_TextureAddressMode addressMode)
{
    const GLint filter = sampleMode == DGL_TSM_POINT ? GL_NEAREST : GL_LINEAR;
    GLint wrap = GL_REPEAT;
    switch (addressMode)
    {
    case DGL_AM_WRAP: wrap = GL_REPEAT; break;
    case DGL_AM_MIRROR: wrap = GL_MIRRORED_REPEAT; break;
    case DGL_AM_CLAMP: wrap = GL_CLAMP_TO_EDGE; break;
    case DGL_AM_MIRROR_ONCE: wrap = GL_MIRRORED_REPEAT; break; // no core-3.3 equivalent
    }
    glSamplerParameteri(g.sampler, GL_TEXTURE_MIN_FILTER, filter);
    glSamplerParameteri(g.sampler, GL_TEXTURE_MAG_FILTER, filter);
    glSamplerParameteri(g.sampler, GL_TEXTURE_WRAP_S, wrap);
    glSamplerParameteri(g.sampler, GL_TEXTURE_WRAP_T, wrap);
}

void DGL_Graphics_SetBlendMode(DGL_BlendMode mode)
{
    switch (mode)
    {
    case DGL_BM_NONE: glDisable(GL_BLEND); return;
    case DGL_BM_BLEND: glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); break;
    case DGL_BM_ADD: glBlendFunc(GL_SRC_ALPHA, GL_ONE); break;
    case DGL_BM_MULTIPLY: glBlendFunc(GL_DST_COLOR, GL_ZERO); break;
    }
    glEnable(GL_BLEND);
}

void DGL_Graphics_SetShaderMode(DGL_PixelShaderMode pixelMode, DGL_VertexShaderMode)
{
    if (pixelMode == DGL_PSM_CUSTOM)
    {
        SetError("custom (HLSL) shaders are not supported; drawing with vertex colors");
        pixelMode = DGL_PSM_COLOR;
    }
    g.pixelMode = pixelMode;
}

void DGL_Graphics_SetCustomPixelShader(const DGL_PixelShader*) {}
void DGL_Graphics_SetCustomVertexShader(const DGL_VertexShader*) {}
void DGL_Graphics_SetTexture(const DGL_Texture* texture) { g.texture = texture; }

// --- Graphics: shaders -------------------------------------------------------

const DGL_PixelShader* DGL_Graphics_LoadPixelShader(const char* filename)
{
    SetError(std::string("HLSL pixel shaders are not supported on OpenGL: ") +
             (filename ? filename : "(null)"));
    return nullptr;
}

const DGL_VertexShader* DGL_Graphics_LoadVertexShader(const char* filename)
{
    SetError(std::string("HLSL vertex shaders are not supported on OpenGL: ") +
             (filename ? filename : "(null)"));
    return nullptr;
}

void DGL_Graphics_FreePixelShader(const DGL_PixelShader** shader)
{
    if (shader)
        *shader = nullptr;
}

void DGL_Graphics_FreeVertexShader(const DGL_VertexShader** shader)
{
    if (shader)
        *shader = nullptr;
}

// --- Graphics: textures ------------------------------------------------------

DGL_Texture* DGL_Graphics_LoadTexture(const char* fileName)
{
    int width = 0, height = 0, channels = 0;
    unsigned char* pixels = stbi_load(fileName, &width, &height, &channels, 4);
    if (!pixels)
    {
        SetError(std::string("could not load texture '") + (fileName ? fileName : "(null)") +
                 "': " + stbi_failure_reason());
        return nullptr;
    }
    DGL_Texture* texture = CreateTexture(pixels, width, height);
    stbi_image_free(pixels);
    return texture;
}

DGL_Texture* DGL_Graphics_LoadTextureFromMemory(const unsigned char* data, int width, int height)
{
    if (!data || width <= 0 || height <= 0)
    {
        SetError("LoadTextureFromMemory: invalid arguments");
        return nullptr;
    }
    return CreateTexture(data, width, height);
}

DGL_Texture* DGL_Graphics_CreateRenderTexture(int width, int height)
{
    DGL_Texture* texture = CreateTexture(nullptr, width, height);
    glGenFramebuffers(1, &texture->framebuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, texture->framebuffer);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture->id, 0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        SetError("render texture framebuffer is incomplete");
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    return texture;
}

void DGL_Graphics_ClearRenderTexture(const DGL_Texture* renderTexture, const DGL_Color* color)
{
    if (!renderTexture || !renderTexture->framebuffer)
        return;
    glBindFramebuffer(GL_FRAMEBUFFER, renderTexture->framebuffer);
    glClearColor(color->r, color->g, color->b, color->a);
    glClear(GL_COLOR_BUFFER_BIT);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void DGL_Graphics_FreeTexture(DGL_Texture** texture)
{
    if (!texture || !*texture)
        return;
    if ((*texture)->framebuffer)
        glDeleteFramebuffers(1, &(*texture)->framebuffer);
    glDeleteTextures(1, &(*texture)->id);
    delete *texture;
    *texture = nullptr;
}

DGL_Vec2 DGL_Graphics_GetTextureSize(DGL_Texture* texture)
{
    if (!texture)
        return {0.0f, 0.0f};
    return {static_cast<float>(texture->width), static_cast<float>(texture->height)};
}

// --- Graphics: meshes --------------------------------------------------------

void DGL_Graphics_StartMesh(void) { g.building.clear(); }

DGL_Mesh* DGL_Graphics_EndMesh(void)
{
    DGL_Mesh* mesh = UploadMesh(g.building, nullptr, 0);
    g.building.clear();
    return mesh;
}

DGL_Mesh* DGL_Graphics_EndMeshIndexed(unsigned* indices, unsigned indexCount)
{
    DGL_Mesh* mesh = UploadMesh(g.building, indices, indexCount);
    g.building.clear();
    return mesh;
}

void DGL_Graphics_AddVertex(const DGL_Vec2* p, const DGL_Color* c, const DGL_Vec2* uv)
{
    g.building.push_back({p->x, p->y, c->r, c->g, c->b, c->a, uv->x, uv->y});
}

void DGL_Graphics_AddTriangle(const DGL_Vec2* p1, const DGL_Color* c1, const DGL_Vec2* uv1,
                              const DGL_Vec2* p2, const DGL_Color* c2, const DGL_Vec2* uv2,
                              const DGL_Vec2* p3, const DGL_Color* c3, const DGL_Vec2* uv3)
{
    DGL_Graphics_AddVertex(p1, c1, uv1);
    DGL_Graphics_AddVertex(p2, c2, uv2);
    DGL_Graphics_AddVertex(p3, c3, uv3);
}

void DGL_Graphics_FreeMesh(DGL_Mesh** mesh)
{
    if (!mesh || !*mesh)
        return;
    if ((*mesh)->ebo)
        glDeleteBuffers(1, &(*mesh)->ebo);
    glDeleteBuffers(1, &(*mesh)->vbo);
    glDeleteVertexArrays(1, &(*mesh)->vao);
    delete *mesh;
    *mesh = nullptr;
}

// --- Graphics: drawing -------------------------------------------------------

void DGL_Graphics_StartDrawing(void)
{
    int fbWidth = 0, fbHeight = 0;
    glfwGetFramebufferSize(g.window, &fbWidth, &fbHeight);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, fbWidth, fbHeight);
    glClearColor(g.background.r, g.background.g, g.background.b, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
}

void DGL_Graphics_FinishDrawing(void)
{
    const double captureStart = glfwGetTime();
    for (const Capture& capture : g_captures)
    {
        if (capture.frame != g.frameCount)
            continue;
        int w = 0, h = 0;
        glfwGetFramebufferSize(g.window, &w, &h);
        std::vector<unsigned char> pixels(static_cast<size_t>(w) * h * 4);
        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        glReadBuffer(GL_BACK);
        glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
        stbi_flip_vertically_on_write(1); // OpenGL rows start at the bottom
        if (stbi_write_png(capture.path.c_str(), w, h, 4, pixels.data(), w * 4))
            std::fprintf(stderr, "[DGL compat] captured frame %u -> %s\n", capture.frame,
                         capture.path.c_str());
        else
            SetError("could not write " + capture.path);
    }
    // Encoding a PNG takes far longer than a frame; keep it out of the next delta time so a
    // capture doesn't change what the program does.
    g.lastFrameTime += glfwGetTime() - captureStart;
    glfwSwapBuffers(g.window);
    if (g_exitAfter && g.frameCount >= g_exitAfter)
        glfwSetWindowShouldClose(g.window, GLFW_TRUE);
}

void DGL_Graphics_DrawMesh(const DGL_Mesh* mesh, DGL_DrawMode mode)
{
    float w = 1.0f, h = 1.0f;
    WindowSize(w, h);
    Draw(mesh, mode, w, h);
}

void DGL_Graphics_DrawMeshToTexture(const DGL_Mesh* mesh, DGL_DrawMode mode,
                                    const DGL_Texture* renderTexture)
{
    if (!renderTexture || !renderTexture->framebuffer)
        return;
    glBindFramebuffer(GL_FRAMEBUFFER, renderTexture->framebuffer);
    glViewport(0, 0, renderTexture->width, renderTexture->height);
    Draw(mesh, mode, static_cast<float>(renderTexture->width),
         static_cast<float>(renderTexture->height));

    int fbWidth = 0, fbHeight = 0;
    glfwGetFramebufferSize(g.window, &fbWidth, &fbHeight);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, fbWidth, fbHeight);
}

// --- Graphics: per-draw constants -------------------------------------------

void DGL_Graphics_SetCB_TransformData(const DGL_Vec2* position, const DGL_Vec2* scale,
                                      float rotationRadians)
{
    // T * R * S, stored column-major.
    const float c = std::cos(rotationRadians);
    const float s = std::sin(rotationRadians);
    g.model = {c * scale->x, s * scale->x, 0, 0, -s * scale->y, c * scale->y, 0, 0, 0, 0, 1, 0,
               position->x, position->y, 0, 1};
}

void DGL_Graphics_SetCB_TransformMatrix(const DGL_Mat4* matrix)
{
    // DGL_Mat4 is row-major (m[row][col]); OpenGL wants column-major.
    for (int row = 0; row < 4; ++row)
        for (int col = 0; col < 4; ++col)
            g.model[col * 4 + row] = matrix->m[row][col];
}

void DGL_Graphics_SetCB_TextureOffset(const DGL_Vec2* textureOffset) { g.texOffset = *textureOffset; }
void DGL_Graphics_SetCB_Alpha(float alpha) { g.alpha = alpha; }
void DGL_Graphics_SetCB_TintColor(const DGL_Color* color) { g.tint = *color; }
void DGL_Graphics_SetCB_ShaderData(float) {}

// --- Input -------------------------------------------------------------------

int DGL_Input_GetMouseScrollDelta(void) { return g.scrollDelta; }
DGL_Vec2 DGL_Input_GetMousePosition(void) { return g.mouse; }

DGL_Vec2 DGL_Input_GetMousePositionDelta(void)
{
    return {g.mouse.x - g.previousMouse.x, g.mouse.y - g.previousMouse.y};
}

BOOL DGL_Input_KeyDown(unsigned char key) { return g.keyDown[key] ? TRUE : FALSE; }

BOOL DGL_Input_KeyTriggered(unsigned char key)
{
    return g.keyDown[key] && !g.keyWasDown[key] ? TRUE : FALSE;
}

BOOL DGL_Input_KeyReleased(unsigned char key)
{
    return !g.keyDown[key] && g.keyWasDown[key] ? TRUE : FALSE;
}

unsigned char DGL_Input_LastKeyTriggered(void) { return g.lastKeyTriggered; }

void DGL_Input_ShowCursor(BOOL show)
{
    glfwSetInputMode(g.window, GLFW_CURSOR, show ? GLFW_CURSOR_NORMAL : GLFW_CURSOR_HIDDEN);
}

// --- Window ------------------------------------------------------------------

BOOL DGL_Window_HasFocus(void)
{
    return g.window && glfwGetWindowAttrib(g.window, GLFW_FOCUSED) ? TRUE : FALSE;
}

DGL_Vec2 DGL_Window_GetMonitorSize(void)
{
    const GLFWvidmode* mode = glfwGetVideoMode(glfwGetPrimaryMonitor());
    if (!mode)
        return {0.0f, 0.0f};
    return {static_cast<float>(mode->width), static_cast<float>(mode->height)};
}

DGL_Vec2 DGL_Window_GetSize(void)
{
    float w = 0.0f, h = 0.0f;
    WindowSize(w, h);
    return {w, h};
}

void DGL_Window_SetSizeAndPosition(int width, int height, int left, int top)
{
    glfwSetWindowSize(g.window, width, height);
    glfwSetWindowPos(g.window, left, top);
}

void DGL_Window_SetSize(int width, int height) { glfwSetWindowSize(g.window, width, height); }
void DGL_Window_SetStyle(unsigned) {} // fullscreen/windowed switching is not implemented

} // extern "C"
