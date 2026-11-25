#pragma once
// A pack contains independently pitched, centered parent grids. The payloads are direct FDK,
// not crops of a resampled overview. Field-by-field little-endian encoding (AArch64 Linux).
struct GridRequest { int n; float fov, pitch; Box b; };
inline std::vector<GridRequest> read_grid_requests(const std::string &path, const Geom &g) {
    const auto raw = small_file(path, 16384);
    if(raw.find('\0')!=std::string::npos) throw std::runtime_error("embedded NUL in grid list");
    std::istringstream stream(raw); std::string line;
    if(!std::getline(stream,line) || line!="CFDK-GRIDS-1") throw std::runtime_error("bad grid list magic");
    std::vector<GridRequest> requests;
    while(std::getline(stream,line)) {
        if(requests.size()>=16) throw std::runtime_error("at most 16 grids");
        std::istringstream in(line); std::string dim, extent, box, junk;
        if(!(in>>dim>>extent>>box) || (in>>junk) || dim.find_first_not_of("0123456789")!=std::string::npos || dim.size()>4 || extent.size()>32 || extent.find_first_not_of("0123456789.eE+-")!=std::string::npos) throw std::runtime_error("expect n fov_mm x,y,z,nx,ny,nz");
        const int n=std::stoi(dim); size_t pos=0; const float fov=std::stof(extent,&pos),pitch=fov/n;
        if(pos!=extent.size() || n<2 || n>4096 || !(std::isfinite(g.D+g.R+g.du+g.dv+g.start+fov) && g.R>fov*.708f && pitch>0)) throw std::runtime_error("invalid grid geometry");
        requests.push_back({n,fov,pitch,parse_box(box,n)});
    }
    if(requests.empty()) throw std::runtime_error("empty grid list");
    return requests;
}
inline void write_multigrid(File &out, const std::vector<GridRequest> &requests) {
    static_assert(sizeof(uint32_t)==4 && sizeof(float)==4,"wire types");
    char magic[]="CFDKMG01"; auto count=uint32_t(requests.size());
    out.transfer(magic,8,true);out.transfer(&count,4,true);
    for(auto r:requests){
        auto n=uint32_t(r.n);out.transfer(&n,4,true);out.transfer(&r.fov,4,true);out.transfer(&r.pitch,4,true);
        auto b=r.b;uint32_t coords[]={uint32_t(b.x),uint32_t(b.y),uint32_t(b.z),uint32_t(b.nx),uint32_t(b.ny),uint32_t(b.nz)};
        out.transfer(coords,sizeof(coords),true);
    }
}
