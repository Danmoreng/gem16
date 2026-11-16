// Offline parent/candidate 12B product-session regression. No timing claims.
#include "gem16/engine.h"
#include <array>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void Require(bool valid,const char* message) {
  if(!valid)throw std::runtime_error(message);
}
void Check(const gem16::Status& status) {
  if(!status.ok())throw std::runtime_error(status.message());
}
void Print(const char* name,const std::vector<std::uint32_t>& ids) {
  std::cout<<'"'<<name<<"\":[";
  for(std::size_t i=0;i<ids.size();++i)std::cout<<(i?",":"")<<ids[i];
  std::cout<<']';
}
gem16::Status Cancel(void* context,std::uint32_t) {
  ++*static_cast<unsigned*>(context);
  return {gem16::StatusCode::kCancelled,"12B audit intentional cancellation"};
}
}
int main(int argc,char** argv) {
  try {
    Require(argc==4,"usage: audit target assistant draft-depth");
    const std::array<std::uint32_t,20> prompt{
      2,105,2364,107,40654,607,7121,506,3658,3730,236761,106,107,
      105,4368,107,100,45518,107,101};
    gem16::ConversationSessionOptions options;
    options.model_directory=argv[1];
    options.mtp_draft_tokens=static_cast<unsigned>(std::stoul(argv[3]));
    Require(options.mtp_draft_tokens==0 || options.mtp_draft_tokens==2,"invalid draft depth");
    if(options.mtp_draft_tokens)options.assistant_model_directory=argv[2];
    options.max_context_tokens=4096;
    options.suppressed_token_ids={1,50,106,258882,258883};
    auto created=gem16::ConversationSession::Create(options);
    Require(created.ok(),"create failed");
    auto session=std::move(created).value();
    auto first=session.Generate(prompt,32);
    Require(first.ok() && first.value().output_token_ids.size()==32,"first generation failed");
    Require(!first.value().token_loop_allocations && !first.value().fallback_count,"first runtime invariant");
    std::vector<std::uint32_t> continuation(prompt.begin(),prompt.end());
    continuation.insert(continuation.end(),first.value().output_token_ids.begin(),first.value().output_token_ids.end());
    continuation.push_back(107);
    auto second=session.Generate(continuation,8);
    Require(second.ok() && second.value().prompt_cached_tokens>0,"cached continuation failed");
    Check(session.Reset());
    auto repeated=session.Generate(prompt,32);
    Require(repeated.ok() && repeated.value().output_token_ids==first.value().output_token_ids,"reset output changed");
    Check(session.Reset());
    unsigned callbacks=0;
    auto cancelled=session.Generate(prompt,16,{},Cancel,&callbacks);
    const bool cancel_ok=!cancelled.ok() && cancelled.status().code()==gem16::StatusCode::kCancelled;
    const bool poisoned=session.is_poisoned();
    // Record cancellation behavior for exact parent comparison. Existing
    // product limitations must not be misattributed to this CUDA candidate.
    Check(session.Reset());
    auto relaunched=session.Generate(prompt,32);
    Require(relaunched.ok() && relaunched.value().output_token_ids==first.value().output_token_ids,"post-cancel reset changed");
    std::cout<<'{';Print("first",first.value().output_token_ids);
    std::cout<<',';Print("continuation",second.value().output_token_ids);
    std::cout<<",\"cached_prompt_tokens\":"<<second.value().prompt_cached_tokens
             <<",\"cancel_returned_cancelled\":"<<(cancel_ok?"true":"false")
             <<",\"cancel_poisoned\":"<<(poisoned?"true":"false")
             <<",\"cancel_callbacks\":"<<callbacks
             <<",\"reset_and_relaunch_exact\":true}\n";
    return 0;
  }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
