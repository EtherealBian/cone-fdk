#include "geometry.cuh"
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char **argv) try {
    if (argc != 4) {
        std::cerr << "usage: cone-fdk views.geom views.f32 volume.f32\n";
        return 2;
    }
    Geometry g{};
    std::ifstream meta(argv[1]);
    std::string magic, extra;
    if (!(meta >> magic >> g.a >> g.v >> g.u >> g.du >> g.dv >> g.R >> g.D) || magic != "POPI1" ||
        g.a != 644 || g.u != 512 || g.v != 512 || g.du != .8f || g.dv != .8f || g.R != 1000 ||
        g.D != 1536)
        throw std::runtime_error("invalid POPI scan header");
    std::vector<Pose> poses(g.a);
    for (auto &p : poses)
        if (!(meta >> p.degrees >> p.offset_x >> p.offset_y) ||
            !(p.degrees >= 0 && p.degrees < 360) || !std::isfinite(p.offset_x) ||
            !std::isfinite(p.offset_y))
            throw std::runtime_error("invalid projection pose");
    if (meta >> extra)
        throw std::runtime_error("extra projection pose");
    std::ifstream in(argv[2], std::ios::binary | std::ios::ate);
    size_t count = size_t(g.a) * g.u * g.v;
    if (in.tellg() != std::streamoff(count * sizeof(float)))
        throw std::runtime_error("wrong projection file size");
    Projections p(count);
    in.seekg(0);
    if (!in.read(reinterpret_cast<char *>(p.data()), count * sizeof(float)))
        throw std::runtime_error("projection read failed");
    Volume out;
    float ms = fdk_reconstruct(p, g, poses, out);
    std::ofstream f(argv[3], std::ios::binary);
    f.write(reinterpret_cast<const char *>(out.data()), out.size() * sizeof(float));
    f.close();
    if (!f)
        throw std::runtime_error("volume write failed");
    std::cout << "backproject_ms=" << ms << '\n';
    return 0;
} catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
}
