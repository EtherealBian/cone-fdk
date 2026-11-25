struct Lib {
    void *h;
    explicit Lib(const char *name) : h(dlopen(name, RTLD_NOW|RTLD_LOCAL)) { if(!h) throw std::runtime_error(std::string("cannot load ")+name+": "+dlerror()); }
    Lib(const Lib&)=delete;
    ~Lib(){dlclose(h);}
    template<class T> T sym(const char *s){auto p=dlsym(h,s);if(!p)throw std::runtime_error(std::string("missing symbol: ")+s);return reinterpret_cast<T>(p);}
};
