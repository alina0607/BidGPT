#pragma once

#include <optional>
#include <string>
#include <vector>

namespace sa
{

// 8-bit RGBA pixels, rows top to bottom.
struct Image
{
    int width = 0;
    int height = 0;
    std::vector<unsigned char> rgba;
};

// PNG (or anything stb_image reads); nothing if the file is missing or not an image.
std::optional<Image> LoadImageRGBA(const std::string& path);

} // namespace sa
