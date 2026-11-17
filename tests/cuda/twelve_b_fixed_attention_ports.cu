// Exact fixed-row local attention comparison, including short and wrapped windows.
#include "cuda/attention/sm120.h"
#include "cuda/layer/reference.h"
#include <cuda_runtime.h>
#include <array>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <iostream>
#include <cstring>
namespace {
using namespace gem16::internal;
void Check(cudaError_t e) {
  if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
void Check(const gem16::Status& s) {
  if (!s.ok()) throw std::runtime_error(s.message());
}
template<class T> struct Buffer {
  std::uint8_t* base = nullptr;
  T* p = nullptr;
  std::size_t n;
  explicit Buffer(std::size_t count): n(count * sizeof(T)) {
    Check(cudaMalloc(&base, n + 512)); p = reinterpret_cast<T*>(base + 256);
    Check(cudaMemset(base, 165, n + 512));
  }
  ~Buffer() { if (base) (void)cudaFree(base); }
  void Put(const std::vector<T>& data) {
    if (data.size() * sizeof(T) != n) throw std::runtime_error("input size");
    Check(cudaMemcpy(p, data.data(), n, cudaMemcpyHostToDevice));
  }
  void Guard() {
    std::array<std::uint8_t, 256> a{}, b{};
    Check(cudaMemcpy(a.data(), base, 256, cudaMemcpyDeviceToHost));
    Check(cudaMemcpy(b.data(), base + 256 + n, 256, cudaMemcpyDeviceToHost));
    if (!std::all_of(a.begin(),a.end(),[](auto x){return x==165;}) ||
        !std::all_of(b.begin(),b.end(),[](auto x){return x==165;}))
      throw std::runtime_error("device guard modified");
  }
  void Save(const std::filesystem::path& path) {
    std::vector<std::uint8_t> bytes(n);
    Check(cudaMemcpy(bytes.data(), p, n, cudaMemcpyDeviceToHost)); Guard();
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(bytes.data()), n);
    if (!f) throw std::runtime_error("output write");
  }
};
}
int main(int argc, char** argv) {
 try {
  if(argc!=3 && argc!=4) throw std::runtime_error("usage: attention-test rows output-directory");
  const unsigned rows=std::stoul(argv[1]);
  if(rows<2 || rows>5) throw std::runtime_error("rows");
  const std::filesystem::path out=argv[2];
  if(!std::filesystem::create_directory(out)) throw std::runtime_error("output exists");
  const auto bytes=[](std::size_t n,unsigned salt){std::vector<std::uint8_t> v(n);for(std::size_t i=0;i<n;++i)v[i]=std::uint8_t(0x20+(i*17+i/251+salt)%32);return v;};
  Buffer<std::uint8_t> k(1024*8*256),v(1024*8*256),sk(rows*8*256),sv(rows*8*256);
  k.Put(bytes(1024*8*256,7));v.Put(bytes(1024*8*256,13));sk.Put(bytes(rows*8*256,23));sv.Put(bytes(rows*8*256,31));
  Buffer<std::uint16_t> ks(1),vs(1);ks.Put({0x3f00});vs.Put({0x3f40});
  Buffer<float> q(rows*16*256),w(rows*(8*16*256+8*16)),result(rows*16*256);
  Buffer<DecodeControl> controls(rows);
  for(unsigned variant=0;variant<3;++variant){
   std::vector<float> queries(rows*16*256);
   for(std::size_t i=0;i<queries.size();++i)queries[i]=variant==0?0.0F:float(int((i*37+i/43)%257)-128)*(variant==1?0.0019F:0.11F);
   q.Put(queries);
   for(std::uint64_t position:{0U,17U,127U,128U,511U,1018U,1020U,1021U,1022U,1023U,1024U,1100U,16384U,17399U}){
    std::vector<DecodeControl> c(rows);for(unsigned r=0;r<rows;++r)c[r].position=position+r;controls.Put(c);
#ifdef GEM16_12B_ATTENTION
    Check(LaunchOnlineAttentionDecodeFp8LocalFixed12BControlledSm120(
#else
    Check(LaunchOnlineAttentionDecodeFp8LocalFixedControlledSm120(
#endif
        q.p,k.p,v.p,sk.p,sv.p,ks.p,vs.p,w.p,result.p,controls.p,rows,1024,nullptr));
    result.Save(out/("v"+std::to_string(variant)+"-p"+std::to_string(position)+".bin"));
    if(argc==4) {
      Buffer<std::uint8_t> ok(1024*8*256),ov(1024*8*256);
      Buffer<float> ow(8*16*256+8*16),ordinary(rows*16*256);
      auto hk=bytes(1024*8*256,7),hv=bytes(1024*8*256,13);
      auto hsk=bytes(rows*8*256,23),hsv=bytes(rows*8*256,31);
      for(unsigned row=0;row<rows;++row) {
        auto slot=(position+row)%1024;
        std::copy_n(hsk.data()+row*8*256,8*256,hk.data()+slot*8*256);
        std::copy_n(hsv.data()+row*8*256,8*256,hv.data()+slot*8*256);
        ok.Put(hk);ov.Put(hv);
        Check(LaunchOnlineAttentionDecodeFp8Sm120(q.p+row*16*256,ok.p,ov.p,ks.p,vs.p,ow.p,ordinary.p+row*16*256,controls.p+row,16,8,256,1024,true,nullptr));
      }
      std::vector<float> a(rows*16*256),b(a.size());
      Check(cudaMemcpy(a.data(),result.p,result.n,cudaMemcpyDeviceToHost));
      Check(cudaMemcpy(b.data(),ordinary.p,ordinary.n,cudaMemcpyDeviceToHost));
      if(std::memcmp(a.data(),b.data(),result.n)!=0) throw std::runtime_error("ordinary mismatch rows="+std::to_string(rows)+" variant="+std::to_string(variant)+" position="+std::to_string(position));
    }
    k.Guard();v.Guard();sk.Guard();sv.Guard();ks.Guard();vs.Guard();q.Guard();w.Guard();controls.Guard();
   }
  }
  std::cout<<"fixed local attention guards passed rows="<<rows<<'\n';
  return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
