// Compare every bit pattern class at every lane/tail against an independent exponent oracle.
#include "finite_sve.hpp"
#include <sys/mman.h>
#include <unistd.h>
#include <cstring>
#include <iostream>
#include <vector>
int main(){
  require_sve();const unsigned lanes=spark_vector_bits()/32; long page=sysconf(_SC_PAGESIZE);
  auto addr=mmap(nullptr,2*page,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);if(addr==MAP_FAILED || mprotect(static_cast<char*>(addr)+page,page,PROT_NONE)) return 2;
  auto end=reinterpret_cast<float*>(static_cast<char*>(addr)+page);unsigned long long ntests=0;uint32_t bits[]={0,0x80000000,1,0x80000001,0x007fffff,0x807fffff,0x00800000,0xff7fffff,0x7f800000,0xff800000,0x7f800001,0xff923456,0x7fc00000,0xffffffff};
  if(!finite_block(end,0))return 3;
  for(unsigned n=1;n<=5*lanes+3;n++){
    float* p=end-n;std::memset(p,0,n*4);
    for(unsigned j=0;j<n;j++){
      for(auto u:bits){std::memcpy(p+j,&u,4); bool expected=((u>>23)&255u)!=255u; if(finite_block(p,n)!=expected) return 4; ++ntests;}
      p[j]=0;
    }
  }
  munmap(addr,2*page);std::cout<<"{\"sve_bits\":"<<lanes*32<<",\"patterns\":"<<ntests<<",\"protected_tail\":true}\n";
}
