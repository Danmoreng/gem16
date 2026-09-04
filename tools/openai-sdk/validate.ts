import assert from 'node:assert/strict';
import os from 'node:os';
import {VERSION} from 'openai/version';
import OpenAI from 'openai';
import type {ChatCompletionMessageParam, ChatCompletionTool} from 'openai/resources/chat/completions';
import type {Response, ResponseInput, ResponseCreateParamsNonStreaming, FunctionTool} from 'openai/resources/responses/responses';

const baseURL = process.argv[2] ?? 'http://127.0.0.1:8080/v1';
const model = process.argv[3] ?? 'gem16';
const client = new OpenAI({baseURL, apiKey:'gem16-local-validation', maxRetries:0, timeout:120_000});
const tool: FunctionTool = {type:'function', name:'lookup', description:'Look up the exact marker for a key. Never guess a marker.',
  parameters:{type:'object',properties:{key:{type:'string',enum:['alpha','beta']}},required:['key'],additionalProperties:false},strict:true};
const chatTools: ChatCompletionTool[] = [{type:'function',function:{name:tool.name,description:tool.description??undefined,parameters:tool.parameters!,strict:true}}];
const markers: Record<string,string> = {alpha:'München-α-731',beta:'東京-β-942'};
const prompt = 'Use lookup for alpha and beta, one call per key. Both lookups are independent. Then report both exact markers.';
type Call = {id:string; name:string; arguments:string};
async function chat(messages:ChatCompletionMessageParam[], stream:boolean) {
  const params = {model,messages,tools:chatTools,parallel_tool_calls:true,reasoning_effort:'none' as const,max_completion_tokens:256};
  if (!stream) {
    const r=await client.chat.completions.create(params);
    assert(r._request_id && r.usage && r.usage.total_tokens>0);
    const c=r.choices[0];
    return {message:c.message as ChatCompletionMessageParam,answer:c.message.content??'',finish:c.finish_reason,
      calls:(c.message.tool_calls??[]).map(c=>{assert(c.type==='function');return {id:c.id,...c.function};})};
  }
  const chunks=await client.chat.completions.create({...params,stream:true,stream_options:{include_usage:true}});
  const calls=new Map<number,Call>(); let answer='',finish='',usage=false;
  for await (const chunk of chunks) {
    if(chunk.usage) usage=chunk.usage.total_tokens>0;
    for(const c of chunk.choices) {
      answer+=c.delta.content??''; if(c.finish_reason)finish=c.finish_reason;
      for(const d of c.delta.tool_calls??[]) {
        const call=calls.get(d.index)??{id:'',name:'',arguments:''};
        if(d.id)call.id=d.id;
        call.name+=d.function?.name??'';call.arguments+=d.function?.arguments??'';calls.set(d.index,call);
      }
    }
  }
  assert(finish && usage);
  const normalized=[...calls.entries()].sort((a,b)=>a[0]-b[0]).map(x=>x[1]);
  const message:ChatCompletionMessageParam={role:'assistant',content:answer||null,
    ...(normalized.length?{tool_calls:normalized.map(c=>({type:'function' as const,id:c.id,function:{name:c.name,arguments:c.arguments}}))}:{})};
  return {message,answer,finish,calls:normalized};
}
async function response(params:Omit<ResponseCreateParamsNonStreaming,'stream'>,stream:boolean):Promise<Response> {
  let r:Response;
  if(!stream){const value=await client.responses.create(params);assert(value._request_id);r=value;}
  else {
    const s=client.responses.stream(params);let last=-1,text='',first='',terminal='';const ids:string[]=[];
    for await(const e of s) {
      assert(e.sequence_number>last);last=e.sequence_number;first ||= e.type;terminal=e.type;
      if(e.type==='response.output_text.delta')text+=e.delta;
      if('response' in e)ids.push(e.response.id);
    }
    r=await s.finalResponse();assert.equal(first,'response.created');assert.equal(terminal,'response.'+r.status);
    assert.equal(text,r.output_text);assert(ids.every(id=>id===r.id));
  }
  assert(r.usage && r.usage.total_tokens>0);assert.equal(r.completed_at===null,r.status!=='completed');return r;
}
async function loop(api:string,stream:boolean,multi=true) {
  const question=multi?prompt:'First use lookup for alpha. Only after receiving its result, use lookup for beta. Then report both exact markers.';
  const history:ChatCompletionMessageParam[]=[{role:'user',content:question}];let input:ResponseInput=[{role:'user',content:question}];
  let previous:string|undefined;const ids:string[]=[];const seen=new Set<string>();let parallel=false;
  for(let turn=0;turn<5;turn++) {
    let calls:Call[],answer:string;
    if(api==='chat') {
      const r=await chat(history,stream);history.push(r.message);calls=r.calls;answer=r.answer;
      assert.equal(r.finish,calls.length?'tool_calls':'stop');
    } else {
      const r=await response({model,input,tools:[tool],parallel_tool_calls:true,reasoning:{effort:'none'},max_output_tokens:256,
        ...(previous?{previous_response_id:previous}:{})},stream);
      assert.equal(r.status,'completed');previous=r.id;ids.push(r.id);answer=r.output_text;input=[];
      calls=r.output.filter(c=>c.type==='function_call').map(c=>({id:c.call_id,name:c.name,arguments:c.arguments}));
    }
    parallel ||= calls.length>1;
    if(!calls.length) {
      if(!multi)assert(turn>=2,'sequential tool rounds were not observed');
      assert.deepEqual([...seen].sort(),['alpha','beta']);assert(Object.values(markers).every(m=>answer.includes(m)),answer);
      if(api==='responses')await assert.rejects(client.responses.create({model,input:'Reject this stale branch.',previous_response_id:ids[0]}),OpenAI.NotFoundError);
      return {turns:turn+1,parallel_calls_observed:parallel,keys:[...seen].sort(),answer};
    }
    assert.equal(new Set(calls.map(c=>c.id)).size,calls.length);
    for(const c of calls) {
      const args=JSON.parse(c.arguments);assert(c.id && c.name==='lookup');assert.deepEqual(Object.keys(args),['key']);assert(args.key in markers);
      seen.add(args.key);const output=JSON.stringify({notes:Array(128).fill('irrelevant padding'),marker:markers[args.key]});
      if(api==='chat')history.push({role:'tool',tool_call_id:c.id,content:output});
      else input.push({type:'function_call_output',call_id:c.id,output});
    }
  }
  throw Error('tool loop exceeded five turns');
}
async function parallelHistory(api:string,stream:boolean) {
  const calls=Object.keys(markers).map(key=>({id:'call_'+key,type:'function' as const,function:{name:'lookup',arguments:JSON.stringify({key})}}));
  let answer:string;
  if(api==='chat') {
    const history:ChatCompletionMessageParam[]=[{role:'user',content:prompt},{role:'assistant',content:null,tool_calls:calls},
      ...calls.map((c,i)=>({role:'tool' as const,tool_call_id:c.id,content:Object.values(markers)[i]}))];
    const r=await chat(history,stream);assert.equal(r.finish,'stop');answer=r.answer;
  }else {
    const input:ResponseInput=[{role:'user',content:prompt},
      ...calls.map(c=>({type:'function_call' as const,call_id:c.id,...c.function})),
      ...calls.map((c,i)=>({type:'function_call_output' as const,call_id:c.id,output:Object.values(markers)[i]}))];
    const r=await response({model,input,tools:[tool],reasoning:{effort:'none'},max_output_tokens:256},stream);
    assert.equal(r.status,'completed');answer=r.output_text;
  }
  assert(Object.values(markers).every(m=>answer.includes(m)),answer);
  return {fixture_call_ids:calls.map(c=>c.id),answer};
}
const cases:Record<string,unknown>[]=[];
async function run(name:string,fn:()=>Promise<unknown>){try{cases.push({name,status:'passed',detail:await fn()});}catch(e){cases.push({name,status:'failed',error:String(e)});}}
for(const api of ['chat','responses'])for(const stream of [false,true]) {
  await run(`${api}-tools-${stream}`,()=>loop(api,stream));
  await run(`${api}-sequential-tools-${stream}`,()=>loop(api,stream,false));
  await run(`${api}-parallel-history-${stream}`,()=>parallelHistory(api,stream));
  await run(`${api}-output-limit-${stream}`,async()=>{
    if(api==='responses') {
      const r=await response({model,input:'Write a long story.',reasoning:{effort:'none'},max_output_tokens:1},stream);
      assert.equal(r.status,'incomplete');assert.equal(r.incomplete_details?.reason,'max_output_tokens');
    } else {
      const p={model,messages:[{role:'user' as const,content:'Write a long story.'}],max_completion_tokens:1,reasoning_effort:'none' as const};
      if(stream){const s=await client.chat.completions.create({...p,stream:true});const finishes=[];
        for await(const c of s)for(const choice of c.choices)if(choice.finish_reason)finishes.push(choice.finish_reason);
        assert.deepEqual(finishes,['length']);
      }else assert.equal((await client.chat.completions.create(p)).choices[0].finish_reason,'length');
    }
    return {expected:api==='responses'?'response.incomplete':'length'};
  });
}
for(const api of ['chat','responses'])await run(`${api}-unsupported`,async()=>{
  try {
    if(api==='chat')await client.chat.completions.create({model,messages:[{role:'user',content:'Hello'}],temperature:0.5});
    else await client.responses.create({model,input:'Hello',store:false});
  }catch(e){assert(e instanceof OpenAI.BadRequestError);assert(e.requestID);return {http_status:e.status,error:e.error};}
  throw Error('unsupported field accepted');
});
const status=cases.every(c=>c.status==='passed')?'passed':'failed';
console.log(JSON.stringify({sdk:'openai-node',version:VERSION,platform:os.platform(),status,cases},null,2));
process.exitCode=status==='passed'?0:1;
