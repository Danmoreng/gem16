// AP1 test-only diagnostic interposer: read existing chain proposals after completion.
// No graph mutation, no per-token work, no device allocation. Disk I/O at exit.
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "cuda/mtp/verify.h"
namespace {
constexpr std::size_t kCapacity=2U*17519U;
std::array<unsigned,kCapacity> proposals{};
gem16::internal::MtpChainResult result{};
const void* device_result=nullptr;
const void* device_proposals=nullptr;
char output_path[4096]{};
bool captured=false;
int copy_error=-1;
using Copy=cudaError_t(*)(void*,const void*,size_t,cudaMemcpyKind);
Copy real_copy=nullptr;
void save() {
 if(output_path[0]==0) return;
 FILE* file=std::fopen(output_path,"wx");if(!file)return;
 std::fprintf(file,"{\"diagnostic_only\":true,\"captured\":%s,\"copy_error\":%d,\"output_count\":%llu,\"group_count\":%llu,\"proposed_count\":%llu,\"accepted_count\":%llu,\"ordinary_tail_count\":%llu,\"proposed_token_ids\":[",captured?"true":"false",copy_error,(unsigned long long)result.output_count,(unsigned long long)result.group_count,(unsigned long long)result.proposed_count,(unsigned long long)result.accepted_count,(unsigned long long)result.ordinary_tail_count);
 if(captured && copy_error==0) for(std::size_t i=0;i<result.proposed_count;++i)std::fprintf(file,"%s%u",i?",":"",proposals[i]);
 std::fputs("]}\n",file);std::fclose(file);
}
__attribute__((constructor)) void initialize() {
 const char* path=std::getenv("GEM16_AUDIT_PROPOSALS_JSON");
 if(path && std::strlen(path)<sizeof(output_path)){std::strcpy(output_path,path);std::atexit(save);}
 real_copy=reinterpret_cast<Copy>(dlsym(RTLD_NEXT,"cudaMemcpy"));
}
void inspect(cudaGraph_t graph,unsigned depth=0) {
 if(!output_path[0] || depth>16)return;
 std::size_t count=0;if(cudaGraphGetNodes(graph,nullptr,&count)!=cudaSuccess || count>100000)return;
 std::vector<cudaGraphNode_t> nodes(count);if(cudaGraphGetNodes(graph,nodes.data(),&count)!=cudaSuccess)return;
 for(auto node:nodes){
  cudaGraphNodeType type;if(cudaGraphNodeGetType(node,&type)!=cudaSuccess)continue;
  if(type==cudaGraphNodeTypeGraph){cudaGraph_t child;if(cudaGraphChildGraphNodeGetGraph(node,&child)==cudaSuccess)inspect(child,depth+1);}
  else if(type==cudaGraphNodeTypeConditional){cudaGraphNodeParams p{};if(cudaGraphNodeGetParams(node,&p)==cudaSuccess)for(unsigned i=0;i<p.conditional.size;++i)inspect(p.conditional.phGraph_out[i],depth+1);}
  else if(type==cudaGraphNodeTypeKernel){
   cudaKernelNodeParams p{};const char* name=nullptr;
   if(cudaGraphKernelNodeGetParams(node,&p)==cudaSuccess && p.kernelParams && cudaFuncGetName(&name,p.func)==cudaSuccess && name && std::strstr(name,"AdvanceMtpD2ChainKernel")){
    std::memcpy(&device_result,p.kernelParams[1],sizeof(device_result));
    std::memcpy(&device_proposals,p.kernelParams[3],sizeof(device_proposals));
   }
  }
 }
}
}
extern "C" cudaError_t cudaGraphInstantiate(cudaGraphExec_t* executable,cudaGraph_t graph,unsigned long long flags){
 static auto real=reinterpret_cast<decltype(&cudaGraphInstantiate)>(dlsym(RTLD_NEXT,"cudaGraphInstantiate"));
 inspect(graph);return real(executable,graph,flags);
}
extern "C" cudaError_t cudaMemcpy(void* dst,const void* src,std::size_t bytes,cudaMemcpyKind kind){
 cudaError_t status=real_copy(dst,src,bytes,kind);
 if(status==cudaSuccess && !captured && device_result && src==device_result && bytes==sizeof(result) && kind==cudaMemcpyDeviceToHost){
  std::memcpy(&result,dst,sizeof(result));
  if(result.proposed_count<=kCapacity && device_proposals){
   copy_error=static_cast<int>(real_copy(proposals.data(),device_proposals,result.proposed_count*sizeof(unsigned),cudaMemcpyDeviceToHost));captured=true;
  }
 }
 return status;
}
