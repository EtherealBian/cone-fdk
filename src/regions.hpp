#pragma once
#include "files.hpp"
#include <cstdint>
#include <sstream>

struct Box {
    int x, y, z, nx, ny, nz;
    size_t plane() const {
        return size_t(nx) * ny;
    }
    size_t voxels() const {
        return plane() * nz;
    }
};
inline Box parse_box(const std::string &text, int n) {
    std::istringstream stream(text);
    int v[6];
    for (int i = 0; i < 6; ++i) {
        std::string s;
        if (!std::getline(stream, s, ',') || s.empty() || s.size() > 4 ||
            s.find_first_not_of("0123456789") != std::string::npos)
            throw std::runtime_error("expected x,y,z,nx,ny,nz (six unsigned grid integers)");
        v[i] = std::stoi(s);
        if (v[i] > n || (i >= 3 && !v[i]))
            throw std::runtime_error("ROI coordinate or dimension out of range");
    }
    if (!stream.eof() || v[0] + v[3] > n || v[1] + v[4] > n || v[2] + v[5] > n)
        throw std::runtime_error("ROI outside the reference grid or extra fields");
    return {v[0], v[1], v[2], v[3], v[4], v[5]};
}
inline std::vector<Box> read_boxes(const std::string &path, int n) {
    auto text = small_file(path, 65536);
    if (text.compare(0, 3, "\xef\xbb\xbf") == 0)
        text.erase(0, 3);
    std::istringstream input(text);
    std::vector<Box> boxes;
    std::string line;
    int number = 0;
    while (std::getline(input, line)) {
        ++number;
        auto a = line.find_first_not_of(" \t\r");
        if (a == std::string::npos || line[a] == '#')
            continue;
        if (boxes.size() == 256)
            throw std::runtime_error("--roi-list accepts at most 256 boxes");
        try {
            boxes.push_back(parse_box(line.substr(a, line.find_last_not_of(" \t\r") - a + 1), n));
        } catch (const std::exception &e) {
            throw std::runtime_error("--roi-list line " + std::to_string(number) + ": " + e.what());
        }
    }
    if (boxes.empty())
        throw std::runtime_error("--roi-list is empty");
    return boxes;
}

// Each pass shares its filtered view batches across all resident tiles. Keep
// output in file order. Split only boxes larger than the entire volume arena.
// Lazy enumeration bounds host planner storage even for thousands of Z slabs.
struct PassPlanner {
    const std::vector<Box> &boxes;
    size_t capacity, index = 0;
    int z = 0;
    PassPlanner(const std::vector<Box> &b, size_t c) : boxes(b), capacity(c) {}
    size_t next(std::vector<Box> &tiles) {
        tiles.clear();
        size_t used = 0;
        while (index < boxes.size()) {
            auto t = boxes[index];
            t.z += z;
            t.nz -= z;
            if (used && t.voxels() > capacity - used)
                break;
            t.nz = int(std::min<size_t>(t.nz, capacity / t.plane()));
            if (!t.nz)
                throw std::runtime_error("internal error: arena cannot fit one plane");
            tiles.push_back(t);
            used += t.voxels();
            z += t.nz;
            if (z == boxes[index].nz) {
                ++index;
                z = 0;
            } else
                break;
        }
        return used;
    }
};
// CFDKROI1 is little endian on the supported Windows x64 ABI. Explicit fields
// only (no struct padding). The first FP32 payload starts at 20 + 24*count.
inline size_t pack_header_bytes(size_t count) {
    return 20 + 24 * count;
}
inline void write_pack_header(File &out, const std::vector<Box> &boxes, int n, float pitch) {
    char magic[] = "CFDKROI1";
    uint32_t grid = n, count = uint32_t(boxes.size());
    out.transfer(magic, 8, true);
    out.transfer(&grid, 4, true);
    out.transfer(&pitch, 4, true);
    out.transfer(&count, 4, true);
    for (auto b : boxes) {
        uint32_t coords[] = {uint32_t(b.x),  uint32_t(b.y),  uint32_t(b.z),
                             uint32_t(b.nx), uint32_t(b.ny), uint32_t(b.nz)};
        out.transfer(coords, sizeof(coords), true);
    }
}
