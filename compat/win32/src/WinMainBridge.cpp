// Windows programs start at wWinMain; everywhere else the C runtime calls main.
#include <Windows.h>

int APIENTRY wWinMain(HINSTANCE instance, HINSTANCE prevInstance, LPWSTR cmdLine, int cmdShow);

int main()
{
    wchar_t emptyCmdLine[] = L"";
    return wWinMain(nullptr, nullptr, emptyCmdLine, SW_SHOW);
}
