#include "OpenGLBackend.h"

// glad ships with GLFW's sources; one loader for Windows, macOS and Linux.
#ifndef SA_THIRD_PARTY_FROM_COMPAT
#define GLAD_GL_IMPLEMENTATION
#endif
#include <glad/gl.h>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#ifndef SA_THIRD_PARTY_FROM_COMPAT
#define STB_IMAGE_WRITE_IMPLEMENTATION
#endif
#include <stb_image_write.h>

#include "../Image.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>

namespace sa
{

namespace
{

constexpr const char* kVertexShader = R"(#version 330 core
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aUV;
layout(location = 2) in vec4 aColor;
uniform vec2 uScreenSize;
out vec2 vUV;
out vec4 vColor;
void main()
{
    // Pixels (top-left origin, y down) to normalized device coordinates.
    vec2 ndc = vec2(aPos.x / uScreenSize.x * 2.0 - 1.0, 1.0 - aPos.y / uScreenSize.y * 2.0);
    gl_Position = vec4(ndc, 0.0, 1.0);
    vUV = aUV;
    vColor = aColor;
}
)";

constexpr const char* kFragmentShader = R"(#version 330 core
in vec2 vUV;
in vec4 vColor;
uniform sampler2D uTexture;
out vec4 fragColor;
void main()
{
    fragColor = texture(uTexture, vUV) * vColor;
}
)";

constexpr int kGlfwKeys[] = {GLFW_KEY_ESCAPE, GLFW_KEY_ENTER, GLFW_KEY_SPACE, GLFW_KEY_BACKSPACE,
                             GLFW_KEY_C,      GLFW_KEY_LEFT,  GLFW_KEY_RIGHT, GLFW_KEY_UP,
                             GLFW_KEY_DOWN, GLFW_KEY_W, GLFW_KEY_A, GLFW_KEY_S, GLFW_KEY_D,
                             GLFW_KEY_LEFT_SHIFT, GLFW_KEY_J, GLFW_KEY_K};
constexpr const char* kKeyNames[] = {"escape", "enter", "space", "backspace", "c",
                                     "left",   "right", "up",    "down", "w", "a", "s", "d", "shift", "j", "k"};
static_assert(sizeof(kGlfwKeys) / sizeof(kGlfwKeys[0]) == static_cast<int>(Key::K) + 1,
              "kGlfwKeys must list every sa::Key in enum order");

unsigned CompileShader(GLenum type, const char* source)
{
    const unsigned shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);

    GLint ok = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (!ok)
    {
        char log[1024];
        glGetShaderInfoLog(shader, sizeof(log), nullptr, log);
        std::cerr << "Shader compile error: " << log << '\n';
        glDeleteShader(shader);
        return 0;
    }
    return shader;
}

unsigned LinkProgram()
{
    const unsigned vs = CompileShader(GL_VERTEX_SHADER, kVertexShader);
    const unsigned fs = CompileShader(GL_FRAGMENT_SHADER, kFragmentShader);
    if (!vs || !fs)
        return 0;

    const unsigned program = glCreateProgram();
    glAttachShader(program, vs);
    glAttachShader(program, fs);
    glLinkProgram(program);
    glDeleteShader(vs);
    glDeleteShader(fs);

    GLint ok = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &ok);
    if (!ok)
    {
        char log[1024];
        glGetProgramInfoLog(program, sizeof(log), nullptr, log);
        std::cerr << "Shader link error: " << log << '\n';
        glDeleteProgram(program);
        return 0;
    }
    return program;
}

void CharCallback(GLFWwindow* window, unsigned codepoint)
{
    static_cast<OpenGLBackend*>(glfwGetWindowUserPointer(window))->OnChar(codepoint);
}

} // namespace

void OpenGLBackend::OnChar(unsigned codepoint)
{
    if (codepoint >= 32 && codepoint < 127)
        typing_ += static_cast<char>(codepoint);
}

bool OpenGLBackend::Init(const WindowDesc& desc)
{
    if (const char* exitAfter = std::getenv("SA_EXIT_AFTER"))
        exitAfter_ = static_cast<unsigned>(std::strtoul(exitAfter, nullptr, 10));
    if (const char* spec = std::getenv("SA_CAPTURE"))
    {
        std::stringstream list(spec);
        std::string item;
        while (std::getline(list, item, ','))
        {
            const size_t at = item.rfind('@');
            if (at != std::string::npos)
                captures_.push_back({item.substr(0, at),
                                     static_cast<unsigned>(std::strtoul(item.c_str() + at + 1, nullptr, 10))});
        }
    }
    if (const char* spec = std::getenv("SA_KEYS"))
    {
        // name@first:last, e.g. right@30:300. This drives the real input path for recorded animation QA.
        std::stringstream list(spec);
        std::string item;
        while (std::getline(list, item, ','))
        {
            const size_t at = item.find('@');
            const size_t colon = item.find(':', at == std::string::npos ? 0 : at + 1);
            if (at == std::string::npos || colon == std::string::npos)
                continue;
            for (int key = 0; key < kKeyCount; ++key)
                if (item.substr(0, at) == kKeyNames[key])
                    heldKeys_.push_back({key,
                                         static_cast<unsigned>(std::strtoul(item.c_str() + at + 1, nullptr, 10)),
                                         static_cast<unsigned>(std::strtoul(item.c_str() + colon + 1, nullptr, 10))});
        }
    }

    if (const char* spec = std::getenv("SA_MOUSE"))
    {
        // x:y:down@frame keyframes in frame order, e.g. 300:200:0@60,300:200:1@62,420:260:1@90,420:260:0@91: the
        // cursor moves in a straight line from one keyframe to the next, the button held as the earlier one says.
        // From the first keyframe on the real mouse is ignored, so a recorded drawing session runs the same way
        // every time through the real input path.
        std::stringstream list(spec);
        std::string item;
        while (std::getline(list, item, ','))
        {
            float x = 0, y = 0;
            int down = 0;
            unsigned frame = 0;
            if (std::sscanf(item.c_str(), "%f:%f:%d@%u", &x, &y, &down, &frame) == 4)
                mouseKeys_.push_back({{x, y}, down != 0, frame});
        }
    }

    if (!glfwInit())
    {
        std::cerr << "glfwInit failed\n";
        return false;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE); // required on macOS
#ifdef __APPLE__
    // QA captures use logical resolution: Retina PNG encoding otherwise stalls the simulation.
    // Ordinary play still uses the display's native Retina framebuffer.
    if (!captures_.empty())
        glfwWindowHint(GLFW_COCOA_RETINA_FRAMEBUFFER, GLFW_FALSE);
#endif

    window_ = glfwCreateWindow(desc.width, desc.height, desc.title.c_str(), nullptr, nullptr);
    if (!window_)
    {
        std::cerr << "glfwCreateWindow failed\n";
        glfwTerminate();
        return false;
    }
    glfwMakeContextCurrent(window_);
    glfwSwapInterval(1);
    glfwSetWindowUserPointer(window_, this);
    glfwSetCharCallback(window_, CharCallback);

    if (!gladLoadGL(glfwGetProcAddress))
    {
        std::cerr << "Failed to load OpenGL 3.3 functions\n";
        return false;
    }

    program_ = LinkProgram();
    if (!program_)
        return false;
    screenSizeLocation_ = glGetUniformLocation(program_, "uScreenSize");

    glGenVertexArrays(1, &vao_);
    glGenBuffers(1, &vbo_);
    glBindVertexArray(vao_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                          reinterpret_cast<void*>(offsetof(Vertex, x)));
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                          reinterpret_cast<void*>(offsetof(Vertex, u)));
    glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                          reinterpret_cast<void*>(offsetof(Vertex, r)));
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glEnableVertexAttribArray(2);

    const unsigned char white[4] = {255, 255, 255, 255};
    whiteTexture_ = MakeGLTexture(white, 1, 1);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    return true;
}

void OpenGLBackend::Shutdown()
{
    if (!window_)
        return;
    for (const Texture& t : textures_)
        glDeleteTextures(1, &t.gl);
    glDeleteTextures(1, &whiteTexture_);
    glDeleteBuffers(1, &vbo_);
    glDeleteVertexArrays(1, &vao_);
    glDeleteProgram(program_);
    glfwDestroyWindow(window_);
    glfwTerminate();
    window_ = nullptr;
}

bool OpenGLBackend::BeginFrame(Color clear)
{
    glfwPollEvents();
    if (glfwWindowShouldClose(window_))
        return false;
    typed_ = std::move(typing_);
    typing_.clear();
    ++frame_;

    for (int i = 0; i < kKeyCount; ++i)
    {
        const bool scripted = std::any_of(heldKeys_.begin(), heldKeys_.end(), [this, i](const HeldKey& held) {
            return held.key == i && frame_ >= held.first && frame_ <= held.last;
        });
        const bool down = glfwGetKey(window_, kGlfwKeys[i]) == GLFW_PRESS || scripted ||
                          (i == static_cast<int>(Key::Shift) && glfwGetKey(window_, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS);
        keyPressed_[i] = down && !keyDown_[i];
        keyDown_[i] = down;
    }

    // Framebuffer size differs from window size on HiDPI displays.
    int fbWidth = 0, fbHeight = 0;
    glfwGetFramebufferSize(window_, &fbWidth, &fbHeight);
    glViewport(0, 0, fbWidth, fbHeight);
    glClearColor(clear.r, clear.g, clear.b, clear.a);
    glClear(GL_COLOR_BUFFER_BIT);

    vertices_.clear();
    batches_.clear();
    return true;
}

void OpenGLBackend::EndFrame()
{
    if (!vertices_.empty())
    {
        int width = 0, height = 0;
        glfwGetWindowSize(window_, &width, &height);

        glUseProgram(program_);
        glUniform1i(glGetUniformLocation(program_, "uTexture"), 0);
        glUniform2f(screenSizeLocation_, static_cast<float>(width), static_cast<float>(height));
        glBindVertexArray(vao_);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_);
        glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(vertices_.size() * sizeof(Vertex)),
                     vertices_.data(), GL_STREAM_DRAW);
        glActiveTexture(GL_TEXTURE0);
        // One draw call per run of quads that share a texture, in submission order.
        for (const Batch& batch : batches_)
        {
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, batch.texture);
            glDrawArrays(GL_TRIANGLES, static_cast<GLint>(batch.first), static_cast<GLsizei>(batch.count));
        }
        glActiveTexture(GL_TEXTURE0);
    }
    for (const Capture& capture : captures_)
    {
        if (capture.frame != frame_)
            continue;
        int w = 0, h = 0;
        glfwGetFramebufferSize(window_, &w, &h);
        std::vector<unsigned char> pixels(static_cast<size_t>(w) * h * 4);
        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
        stbi_flip_vertically_on_write(1);
        stbi_write_png(capture.path.c_str(), w, h, 4, pixels.data(), w * 4);
        std::cerr << "captured frame " << frame_ << " -> " << capture.path << '\n';
    }
    glfwSwapBuffers(window_);
    if (exitAfter_ && frame_ >= exitAfter_)
        glfwSetWindowShouldClose(window_, GLFW_TRUE);
}

bool OpenGLBackend::ScriptedMouse(Vec2& pos, bool& down) const
{
    if (mouseKeys_.empty() || frame_ < mouseKeys_.front().frame)
        return false;
    size_t i = 0;
    while (i + 1 < mouseKeys_.size() && mouseKeys_[i + 1].frame <= frame_)
        ++i;
    const MouseKey& a = mouseKeys_[i];
    pos = a.pos;
    down = a.down;
    if (i + 1 < mouseKeys_.size())
    {
        const MouseKey& b = mouseKeys_[i + 1];
        const float t = static_cast<float>(frame_ - a.frame) / static_cast<float>(b.frame - a.frame);
        pos = {a.pos.x + (b.pos.x - a.pos.x) * t, a.pos.y + (b.pos.y - a.pos.y) * t};
    }
    return true;
}

Vec2 OpenGLBackend::MousePosition() const
{
    Vec2 scripted;
    bool down = false;
    if (ScriptedMouse(scripted, down))
        return scripted;
    double x = 0.0, y = 0.0;
    glfwGetCursorPos(window_, &x, &y);
    return {static_cast<float>(x), static_cast<float>(y)};
}

bool OpenGLBackend::MouseDown(MouseButton button) const
{
    Vec2 pos;
    bool down = false;
    if (ScriptedMouse(pos, down))
        return button == MouseButton::Left && down;
    const int glfwButton =
        button == MouseButton::Left ? GLFW_MOUSE_BUTTON_LEFT : GLFW_MOUSE_BUTTON_RIGHT;
    return glfwGetMouseButton(window_, glfwButton) == GLFW_PRESS;
}

bool OpenGLBackend::KeyPressed(Key key) const { return keyPressed_[static_cast<int>(key)]; }

bool OpenGLBackend::KeyDown(Key key) const { return keyDown_[static_cast<int>(key)]; }

void OpenGLBackend::DrawLine(Vec2 from, Vec2 to, float thickness, Color color)
{
    Vec2 d = to - from;
    const float length = std::sqrt(d.x * d.x + d.y * d.y);
    if (length < 1e-4f)
        d = {1.0f, 0.0f}; // a click without movement still leaves a dot
    else
        d = d * (1.0f / length);

    // Extend by half the thickness at both ends so consecutive segments overlap.
    const float h = thickness * 0.5f;
    const Vec2 along = d * h;
    const Vec2 normal{-d.y * h, d.x * h};
    const Vec2 a = from - along;
    const Vec2 b = to + along;
    PushQuad({a + normal, b + normal, b - normal, a - normal}, color, whiteTexture_, {});
}

void OpenGLBackend::DrawRect(Vec2 topLeft, Vec2 size, Color color)
{
    PushQuad({topLeft, {topLeft.x + size.x, topLeft.y}, topLeft + size, {topLeft.x, topLeft.y + size.y}},
             color, whiteTexture_, {});
}

TextureId OpenGLBackend::LoadTexture(const std::string& path, TextureFilter filter)
{
    const auto image = LoadImageRGBA(path);
    if (!image)
    {
        std::cerr << "could not load texture " << path << '\n';
        return 0;
    }
    const unsigned gl = MakeGLTexture(image->rgba.data(), image->width, image->height, filter);
    textures_.push_back({gl, {static_cast<float>(image->width), static_cast<float>(image->height)}});
    return static_cast<TextureId>(textures_.size());
}

TextureId OpenGLBackend::CreateTexture(int width, int height, const unsigned char* rgba, TextureFilter filter)
{
    if (width <= 0 || height <= 0 || !rgba)
        return 0;
    const unsigned gl = MakeGLTexture(rgba, width, height, filter);
    textures_.push_back({gl, {static_cast<float>(width), static_cast<float>(height)}});
    return static_cast<TextureId>(textures_.size());
}

void OpenGLBackend::UpdateTexture(TextureId texture, const unsigned char* rgba)
{
    if (texture < 1 || texture > textures_.size() || !rgba)
        return;
    const Texture& t = textures_[texture - 1];
    glBindTexture(GL_TEXTURE_2D, t.gl);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, static_cast<int>(t.size.x), static_cast<int>(t.size.y), GL_RGBA,
                    GL_UNSIGNED_BYTE, rgba);
}

Vec2 OpenGLBackend::TextureSize(TextureId texture) const
{
    return texture >= 1 && texture <= textures_.size() ? textures_[texture - 1].size : Vec2{};
}

void OpenGLBackend::DrawSprite(TextureId texture, UVRect source, Vec2 topLeft, Vec2 size, float alpha)
{
    if (texture < 1 || texture > textures_.size())
        return;
    PushQuad({topLeft, {topLeft.x + size.x, topLeft.y}, topLeft + size, {topLeft.x, topLeft.y + size.y}},
             {1.0f, 1.0f, 1.0f, alpha}, textures_[texture - 1].gl, source);
}

void OpenGLBackend::DrawSpriteFilled(TextureId texture, UVRect source, Vec2 topLeft, Vec2 size, Color color)
{
    if (texture < 1 || texture > textures_.size())
        return;
    // The shader multiplies texture by vertex color, so a white texel takes the color as it is.
    PushQuad({topLeft, {topLeft.x + size.x, topLeft.y}, topLeft + size, {topLeft.x, topLeft.y + size.y}}, color,
             textures_[texture - 1].gl, source);
}

unsigned OpenGLBackend::MakeGLTexture(const unsigned char* rgba, int width, int height, TextureFilter filter)
{
    unsigned gl = 0;
    glGenTextures(1, &gl);
    glBindTexture(GL_TEXTURE_2D, gl);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, rgba);
    // Pixel art: no smoothing, and no bleeding from the next sprite-sheet cell at the edges. Smooth sheets
    // leave transparent gaps between cells instead.
    const GLint gl_filter = filter == TextureFilter::Smooth ? GL_LINEAR : GL_NEAREST;
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, gl_filter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, gl_filter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    return gl;
}

// Corners clockwise from the top-left; the texture's v = 0 row maps to the top edge.
void OpenGLBackend::PushQuad(const std::array<Vec2, 4>& corners, Color c, unsigned texture, UVRect uv)
{
    if (batches_.empty() || batches_.back().texture != texture)
        batches_.push_back({texture, vertices_.size(), 0});
    const float u0 = uv.offset.x, v0 = uv.offset.y, u1 = u0 + uv.size.x, v1 = v0 + uv.size.y;
    const std::array<Vec2, 4> uvs{{{u0, v0}, {u1, v0}, {u1, v1}, {u0, v1}}};
    for (int i : {0, 1, 2, 0, 2, 3})
        vertices_.push_back({corners[i].x, corners[i].y, uvs[i].x, uvs[i].y, c.r, c.g, c.b, c.a});
    batches_.back().count += 6;
}

} // namespace sa
