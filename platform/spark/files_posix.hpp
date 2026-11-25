#pragma once
#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>
inline std::runtime_error ioerr(const std::string &s) { return std::runtime_error(s + ": " + std::strerror(errno)); }
inline std::string absolute(const std::string &p) { if(p.empty()) throw std::runtime_error("empty path"); return p; }
inline bool path_exists(const std::string &p) { struct stat a{}; if(!lstat(p.c_str(), &a)) return true; if(errno==ENOENT) return false; throw ioerr(p); }
inline bool identity(const std::string &p, struct stat &s) { if(!stat(p.c_str(), &s)) return true; if(errno==ENOENT) return false; throw ioerr(p); }
inline bool same_file(const std::string &a, const std::string &b) { struct stat x{}, y{}; return a==b || (identity(a,x) && identity(b,y) && x.st_dev==y.st_dev && x.st_ino==y.st_ino); }
struct File {
 int fd=-1; explicit File(const std::string &p,bool write=false):fd(open(p.c_str(),O_CLOEXEC|(write?(O_WRONLY|O_CREAT|O_EXCL):O_RDONLY),0600)){if(fd<0)throw ioerr("cannot open "+p);}
 ~File(){if(fd>=0)close(fd);} File(const File&)=delete;
 size_t size(){struct stat s{}; if(fstat(fd,&s)||s.st_size<0)throw ioerr("input size unavailable");return size_t(s.st_size);}
 void seek(size_t off){if(lseek(fd,static_cast<off_t>(off),SEEK_SET)<0)throw ioerr("input seek failed");} void rewind(){seek(0);}
 void transfer(void *ptr,size_t n,bool wr){auto p=static_cast<unsigned char*>(ptr);while(n){auto chunk=std::min<size_t>(n,1u<<28);ssize_t done=wr?::write(fd,p,chunk) : ::read(fd,p,chunk);if(done<0 && errno==EINTR)continue;if(done<=0)throw std::runtime_error(wr?"output write failed":"input read failed");n-=size_t(done);p+=done;}}
 void finish(){if(fsync(fd))throw ioerr("output flush failed");int f=fd;fd=-1;if(close(f))throw ioerr("output close failed");}
};
inline std::string small_file(const std::string &p,size_t max){File f(p); auto n=f.size();if(!n||n>max)throw std::runtime_error("invalid metadata or PTX length");std::string s(n,'\0');f.transfer(&s[0],n,false);return s;}
struct Output {
 std::string tmp,dest;std::unique_ptr<File> file;bool published=false;
 explicit Output(const std::string &d):tmp(d+".partial."+std::to_string(getpid())),dest(d),file(new File(tmp,true)){}
 ~Output(){file.reset();if(!published)unlink(tmp.c_str());}
 void publish(){file->finish();if(rename(tmp.c_str(),dest.c_str()))throw ioerr("cannot publish output");published=true;}
};
// link() fails with EEXIST atomically, even if another publisher won after checking.
inline bool publish_fresh(const std::string &tmp,const std::string &dst){if(link(tmp.c_str(),dst.c_str()))return false;if(unlink(tmp.c_str()))throw ioerr("cannot unlink published temporary");return true;}
