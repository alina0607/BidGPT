#include "Image.h"

#ifndef SA_THIRD_PARTY_FROM_COMPAT
#define STB_IMAGE_IMPLEMENTATION
#endif
#include <stb_image.h>

namespace sa
{

std::optional<Image> LoadImageRGBA(const std::string& path)
{
    int width = 0, height = 0, channels = 0;
    unsigned char* pixels = stbi_load(path.c_str(), &width, &height, &channels, 4);
    if (!pixels)
        return std::nullopt;
    Image image{width, height, std::vector<unsigned char>(pixels, pixels + static_cast<size_t>(width) * height * 4)};
    stbi_image_free(pixels);
    return image;
}

} // namespace sa
