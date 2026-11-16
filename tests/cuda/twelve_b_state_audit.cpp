// Diagnostic interposer for pinned full-request binaries. Never used for timing.
// Copy the requested target KV arena just before its original cudaFree; graph
// inspection queries resource attributes without changing any graph or kernel.
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "cuda/mtp/verify.h"

namespace {
void* cache = nullptr;
std::size_t cache_bytes = 0;
bool saved = false;
bool resource_saved = false;
unsigned matches = 0;
const char* prefix = nullptr;
gem16::internal::MtpChainResult chain{};
bool chain_seen=false;
bool proposals_seen=false;
void fail(const char* message) {
  std::fprintf(stderr,"12B state audit: %s\n",message);
  std::abort();
}
__attribute__((constructor)) void initialize() {
  prefix = std::getenv("GEM16_12B_STATE_AUDIT_PREFIX");
  const char* bytes = std::getenv("GEM16_12B_STATE_AUDIT_CACHE_BYTES");
  if (prefix && bytes) {
    char* end=nullptr;
    cache_bytes=std::strtoull(bytes,&end,10);
    if (!end || *end || cache_bytes==0 || cache_bytes>2ULL*1024*1024*1024)
      fail("invalid explicitly requested arena size");
  }
}
void inspect(cudaGraph_t graph, unsigned depth=0) {
  if (!prefix || depth>16 || resource_saved) return;
  std::size_t count=0;
  if (cudaGraphGetNodes(graph,nullptr,&count)!=cudaSuccess || count>100000)
    fail("graph size query failed");
  std::vector<cudaGraphNode_t> nodes(count);
  if (cudaGraphGetNodes(graph,nodes.data(),&count)!=cudaSuccess)
    fail("graph nodes query failed");
  for(auto node:nodes) {
    cudaGraphNodeType type;
    if (cudaGraphNodeGetType(node,&type)!=cudaSuccess) fail("graph type failed");
    if(type==cudaGraphNodeTypeGraph) {
      cudaGraph_t child;
      if(cudaGraphChildGraphNodeGetGraph(node,&child)!=cudaSuccess) fail("child query failed");
      inspect(child,depth+1);
    } else if(type==cudaGraphNodeTypeKernel) {
      cudaKernelNodeParams p{};
      const char* name=nullptr;
      if(cudaGraphKernelNodeGetParams(node,&p)!=cudaSuccess ||
         cudaFuncGetName(&name,p.func)!=cudaSuccess) fail("kernel query failed");
      if(name && std::strstr(name,"TwelveBPostAttentionQuantizeKernel")) {
        cudaFuncAttributes attributes{};
        if(cudaFuncGetAttributes(&attributes,p.func)!=cudaSuccess) fail("attributes failed");
        char path[4096];
        if(std::snprintf(path,sizeof(path),"%s.resources.json",prefix)>=int(sizeof(path)))
          fail("path too long");
        FILE* f=std::fopen(path,"wx");
        if(!f)fail("resource file exists or cannot be written");
        std::fprintf(f,"{\"diagnostic_only\":true,\"sharedSizeBytes\":%zu,\"localSizeBytes\":%zu,\"numRegs\":%d,\"binaryVersion\":%d,\"ptxVersion\":%d}\n",
                     attributes.sharedSizeBytes,attributes.localSizeBytes,
                     attributes.numRegs,attributes.binaryVersion,attributes.ptxVersion);
        if(std::fclose(f))fail("resource close failed");
        resource_saved=true;
      }
    }
  }
}
}
extern "C" cudaError_t cudaMalloc(void** output,std::size_t bytes) {
  static auto real=reinterpret_cast<decltype(&cudaMalloc)>(dlsym(RTLD_NEXT,"cudaMalloc"));
  const auto status=real(output,bytes);
  if(prefix && cache_bytes && status==cudaSuccess && bytes==cache_bytes) {
    if(++matches!=1)fail("ambiguous arena size");
    cache=*output;
  }
  return status;
}
extern "C" cudaError_t cudaFree(void* pointer) {
  static auto real=reinterpret_cast<decltype(&cudaFree)>(dlsym(RTLD_NEXT,"cudaFree"));
  if(prefix && pointer && pointer==cache && !saved) {
    if(cudaDeviceSynchronize()!=cudaSuccess)fail("final synchronization failed");
    char path[4096];
    if(std::snprintf(path,sizeof(path),"%s.kv.bin",prefix)>=int(sizeof(path)))fail("path too long");
    FILE* f=std::fopen(path,"wx");
    if(!f)fail("KV file exists or cannot be written");
    std::array<unsigned char,1024*1024> buffer{};
    for(std::size_t offset=0;offset<cache_bytes;) {
      const auto bytes=std::min(buffer.size(),cache_bytes-offset);
      if(cudaMemcpy(buffer.data(),static_cast<const unsigned char*>(cache)+offset,
                    bytes,cudaMemcpyDeviceToHost)!=cudaSuccess)fail("KV read failed");
      if(std::fwrite(buffer.data(),1,bytes,f)!=bytes)fail("KV write failed");
      offset+=bytes;
    }
    if(std::fclose(f))fail("KV close failed");
    saved=true;
    std::fprintf(stderr,"12B state audit: saved %zu KV bytes\n",cache_bytes);
  }
  return real(pointer);
}
extern "C" cudaError_t cudaMemcpy(void* destination,const void* source,
                                   std::size_t bytes,cudaMemcpyKind kind) {
  static auto real=reinterpret_cast<decltype(&cudaMemcpy)>(dlsym(RTLD_NEXT,"cudaMemcpy"));
  const auto status=real(destination,source,bytes,kind);
  if(!prefix || status!=cudaSuccess || kind!=cudaMemcpyDeviceToHost)return status;
  // The 12B engine already copies these payloads after the chain completes.
  // Observe those successful copies; don't add any read during inference.
  if(bytes==sizeof(chain) && !chain_seen) {
    gem16::internal::MtpChainResult value{};
    std::memcpy(&value,destination,sizeof(value));
    if(value.group_count==507 && value.proposed_count==1014 &&
       value.accepted_count==626 && value.rejected_count==388) {
      chain=value;chain_seen=true;
    }
  } else if(chain_seen && !proposals_seen && bytes==chain.proposed_count*sizeof(std::uint32_t)) {
    char path[4096];
    if(std::snprintf(path,sizeof(path),"%s.chain.json",prefix)>=int(sizeof(path)))fail("path too long");
    FILE* f=std::fopen(path,"wx");
    if(!f)fail("chain file exists or cannot be written");
    std::fprintf(f,"{\"group_count\":%llu,\"proposed_count\":%llu,\"accepted_count\":%llu,\"output_count\":%llu,\"proposed_token_ids\":[",
                 chain.group_count,chain.proposed_count,chain.accepted_count,chain.output_count);
    const auto* ids=static_cast<const std::uint32_t*>(destination);
    for(std::size_t i=0;i<chain.proposed_count;++i)std::fprintf(f,"%s%u",i?",":"",ids[i]);
    std::fputs("]}\n",f);
    if(std::fclose(f))fail("chain close failed");
    proposals_seen=true;
  }
  return status;
}
extern "C" cudaError_t cudaGraphInstantiate(cudaGraphExec_t* executable,
                                            cudaGraph_t graph,
                                            unsigned long long flags) {
  static auto real=reinterpret_cast<decltype(&cudaGraphInstantiate)>(dlsym(RTLD_NEXT,"cudaGraphInstantiate"));
  inspect(graph);
  return real(executable,graph,flags);
}
