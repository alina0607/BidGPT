/* font8x8 (public domain, Daniel Hepper) is a C header whose initializers don't compile as C++,
   so it lives in its own C file behind one function. */
#include "font8x8_basic.h"

const unsigned char* sa_font8x8_glyph(int c)
{
    if (c < 0 || c > 127)
        c = '?';
    return (const unsigned char*)font8x8_basic[c];
}
