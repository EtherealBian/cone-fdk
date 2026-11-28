#include "geometry.cuh"
#include <fstream>
#include <iostream>
#include <string>
#include <stdexcept>
int main(int argc, char **argv) try {
    Geometry g{17, 24, 32, 32, 1, 1, 100, 160, .5f, 0, 1};
    Projections p;
    if (argc == 1) {
        p.resize(size_t(g.a) * g.v * g.u);
        for (size_t i = 0; i < p.size(); ++i)
            p[i] = float((i * 17) % 97) / 97.f;
    } else if (argc == 6) {
        std::ifstream meta(argv[1]);
        std::string magic, extra;
        if (!(meta >> magic >> g.a >> g.v >> g.u >> g.du >> g.dv >> g.R >> g.D >> g.start >>
              g.sign) ||
            magic != "CFDK1" || (meta >> extra))
            throw std::runtime_error("invalid CFDK1 header");
        g.n = std::stoi(argv[3]);
        g.vox = std::stof(argv[4]) / g.n;
        std::ifstream in(argv[2], std::ios::binary | std::ios::ate);
        auto bytes = in.tellg();
        if (bytes < 0 || bytes % 4)
            throw std::runtime_error("invalid float32 projection file");
        p.resize(size_t(bytes) / 4);
        in.seekg(0);
        if (!in.read(reinterpret_cast<char *>(p.data()), bytes))
            throw std::runtime_error("projection read failed");
    } else {
        std::cerr << "usage: cone-fdk [scan.geom scan.f32 grid_n fov_mm out.f32]\n";
        return 2;
    }
    Volume out;
    fdk_reconstruct(p, g, out);
    if (argc == 6) {
        std::ofstream f(argv[5], std::ios::binary);
        f.write(reinterpret_cast<const char *>(out.data()), out.size() * 4);
        f.close();
        if (!f)
            throw std::runtime_error("volume write failed");
    } else
        std::cout << "Dense FDK: " << g.n << "^3, " << g.a << " views; center voxel "
                  << out[out.size() / 2 + g.n * g.n / 2 + g.n / 2] << '\n';
    return 0;
} catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
}
