// Minimal Win32 surface so DGL programs written for Windows compile on macOS and Linux.
// Only types, constants and a handful of no-op functions; no windowing (DGL owns the window).
// Deliberately does NOT define _WINDOWS_, so libraries that sniff it pick their POSIX paths.
// (Case-sensitive file systems also get a generated <windows.h>; see compat/CMakeLists.txt.)

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>

typedef int BOOL;
typedef unsigned char BYTE;
typedef unsigned short WORD;
typedef unsigned long DWORD;
typedef unsigned int UINT;
typedef long LONG;
typedef std::uintptr_t WPARAM;
typedef std::intptr_t LPARAM;
typedef std::intptr_t LRESULT;
typedef wchar_t WCHAR;
typedef WCHAR* LPWSTR;
typedef const WCHAR* LPCWSTR;
typedef char* LPSTR;
typedef const char* LPCSTR;
typedef void* HANDLE;
typedef struct HWND__* HWND;
typedef struct HINSTANCE__* HINSTANCE;
typedef HINSTANCE HMODULE;

#define CALLBACK
#define WINAPI
#define APIENTRY

typedef LRESULT(CALLBACK* WNDPROC)(HWND, UINT, WPARAM, LPARAM);

#ifndef TRUE
#define TRUE 1
#endif
#ifndef FALSE
#define FALSE 0
#endif

#ifndef UNREFERENCED_PARAMETER
#define UNREFERENCED_PARAMETER(p) ((void)(p))
#endif

// Window class / window styles (accepted and ignored by the DGL compat layer).
#define CS_VREDRAW 0x0001
#define CS_HREDRAW 0x0002
#define WS_OVERLAPPED 0x00000000L
#define WS_POPUP 0x80000000L
#define WS_CAPTION 0x00C00000L
#define WS_SYSMENU 0x00080000L
#define WS_THICKFRAME 0x00040000L
#define WS_MINIMIZEBOX 0x00020000L
#define WS_MAXIMIZEBOX 0x00010000L
#define WS_OVERLAPPEDWINDOW                                                                       \
    (WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_THICKFRAME | WS_MINIMIZEBOX | WS_MAXIMIZEBOX)
#define WS_EX_TOPMOST 0x00000008L
#define SW_HIDE 0
#define SW_SHOW 5

// Virtual-key codes (https://learn.microsoft.com/windows/win32/inputdev/virtual-key-codes).
#define VK_LBUTTON 0x01
#define VK_RBUTTON 0x02
#define VK_CANCEL 0x03
#define VK_MBUTTON 0x04
#define VK_BACK 0x08
#define VK_TAB 0x09
#define VK_RETURN 0x0D
#define VK_SHIFT 0x10
#define VK_CONTROL 0x11
#define VK_MENU 0x12
#define VK_PAUSE 0x13
#define VK_CAPITAL 0x14
#define VK_ESCAPE 0x1B
// Window messages the compat layer sends to DGL_SysInitInfo.pWindowsCallback
#define WM_CHAR 0x0102

#define VK_SPACE 0x20
#define VK_PRIOR 0x21
#define VK_NEXT 0x22
#define VK_END 0x23
#define VK_HOME 0x24
#define VK_LEFT 0x25
#define VK_UP 0x26
#define VK_RIGHT 0x27
#define VK_DOWN 0x28
#define VK_INSERT 0x2D
#define VK_DELETE 0x2E
#define VK_NUMPAD0 0x60
#define VK_NUMPAD1 0x61
#define VK_NUMPAD2 0x62
#define VK_NUMPAD3 0x63
#define VK_NUMPAD4 0x64
#define VK_NUMPAD5 0x65
#define VK_NUMPAD6 0x66
#define VK_NUMPAD7 0x67
#define VK_NUMPAD8 0x68
#define VK_NUMPAD9 0x69
#define VK_MULTIPLY 0x6A
#define VK_ADD 0x6B
#define VK_SUBTRACT 0x6D
#define VK_DECIMAL 0x6E
#define VK_DIVIDE 0x6F
#define VK_F1 0x70
#define VK_F2 0x71
#define VK_F3 0x72
#define VK_F4 0x73
#define VK_F5 0x74
#define VK_F6 0x75
#define VK_F7 0x76
#define VK_F8 0x77
#define VK_F9 0x78
#define VK_F10 0x79
#define VK_F11 0x7A
#define VK_F12 0x7B
#define VK_NUMLOCK 0x90
#define VK_SCROLL 0x91
#define VK_LSHIFT 0xA0
#define VK_RSHIFT 0xA1
#define VK_LCONTROL 0xA2
#define VK_RCONTROL 0xA3
#define VK_LMENU 0xA4
#define VK_RMENU 0xA5
#define VK_OEM_1 0xBA
#define VK_OEM_PLUS 0xBB
#define VK_OEM_COMMA 0xBC
#define VK_OEM_MINUS 0xBD
#define VK_OEM_PERIOD 0xBE
#define VK_OEM_2 0xBF
#define VK_OEM_3 0xC0
#define VK_OEM_4 0xDB
#define VK_OEM_5 0xDC
#define VK_OEM_6 0xDD
#define VK_OEM_7 0xDE

inline LRESULT DefWindowProc(HWND, UINT, WPARAM, LPARAM) { return 0; }
inline HMODULE GetModuleHandle(LPCSTR) { return nullptr; }
inline void OutputDebugStringA(LPCSTR message) { std::fputs(message, stderr); }
inline void OutputDebugStringW(LPCWSTR message) { std::fprintf(stderr, "%ls", message); }
