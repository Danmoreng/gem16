#!/usr/bin/env python3
"""Bounded C06 real-SDK tool recovery and explicit unsupported-feature checks."""
import argparse
import json
import sys
import urllib.error
import urllib.request
import openai
from validate_openai_sdk import TOOL, chat, responses


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    p=argparse.ArgumentParser()
    p.add_argument('--base-url', required=True)
    args=p.parse_args()
    client=openai.OpenAI(base_url=args.base_url, api_key='gem16-local-validation', max_retries=0, timeout=90)
    results=[]
    def run(name, function):
        try: results.append({'case':name, 'passed':True, 'detail':function()})
        except Exception as e: results.append({'case':name, 'passed':False, 'error':repr(e)})
    def rejected(route, body):
        request=urllib.request.Request(args.base_url+'/'+route, json.dumps(body).encode(), {'Content-Type':'application/json'})
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                raise AssertionError(f'unsupported request accepted: {response.status}')
        except urllib.error.HTTPError as error:
            detail=json.load(error)
            assert error.code==400 and 'error' in detail, (error.code,detail)
            return detail
    for api in ['chat/completions','responses']:
        base={'model':'gem16','messages':[{'role':'user','content':'Hello'}]} if api.startswith('chat') else {'model':'gem16','input':'Hello'}
        named={'type':'function','function':{'name':'lookup'}} if api.startswith('chat') else {'type':'function','name':'lookup'}
        tool={'type':'function','function':{k:v for k,v in TOOL.items() if k!='type'}} if api.startswith('chat') else TOOL
        for name,fields in [('temperature',{'temperature':.5}),('top_p',{'top_p':.9}),('seed',{'seed':42}),
                            ('required',{'tools':[tool],'tool_choice':'required'}),('named',{'tools':[tool],'tool_choice':named}),
                            ('parallel_false',{'tools':[tool],'parallel_tool_calls':False})]:
            run(api+'/'+name, lambda api=api, fields=fields, base=base:rejected(api,{**base,**fields}))
        bad={**base}
        if api.startswith('chat'):
            bad['messages'] += [{'role':'tool','tool_call_id':'missing','content':'orphan result'}]
        else: bad['input']=[{'type':'function_call_output','call_id':'missing','output':'orphan result'}]
        run(api+'/orphan_tool_id', lambda api=api,bad=bad:rejected(api,bad))
    run('chat/reasoning_content_rejected', lambda:rejected('chat/completions', {'model':'gem16','messages':[
        {'role':'user','content':'Hello'}, {'role':'assistant','content':'Hello','reasoning_content':'unsupported retained reasoning'},
        {'role':'user','content':'Continue'}]}))
    def reasoning_replay():
        r=client.responses.create(model='gem16',input='What is 17 times 19? Give the result.',reasoning={'effort':'low'},max_output_tokens=256)
        assert any(item.type=='reasoning' for item in r.output), 'no actual reasoning item produced'
        try:
            client.responses.create(model='gem16',input=[{'role':'user','content':'What is 17 times 19? Give the result.'},*r.output,
                {'role':'user','content':'Repeat the answer.'}], reasoning={'effort':'low'},max_output_tokens=64)
        except openai.BadRequestError as error:
            return {'generated_response':r.model_dump(),'replay_status':error.status_code,'error':str(error),'qualified':False}
        raise AssertionError('reasoning replay accepted despite declared unsupported boundary')
    run('responses/actual_reasoning_replay_rejected', reasoning_replay)
    marker='Grüße-東京-α-42'
    large_result=('Reference material: ordinary filler, no marker on this line.\n'*250)+'\nExact marker: '+marker
    def recover(api,stream):
        history=[{'role':'user','content':'Use lookup for alpha. If it returns a temporary error, retry the same lookup once. Then output the exact marker from its successful result.'}]
        tools=[TOOL] if api=='responses' else [{'type':'function','function':{k:v for k,v in TOOL.items() if k!='type'}}]
        rounds=[]
        calls_seen=0
        for turn in range(4):
            if api=='chat':
                r=chat(client,'gem16',history,stream,tools=tools)
                history.append(r['message'])
                calls=r['message'].get('tool_calls',[])
                answer=r['message'].get('content') or ''
                rounds.append(r)
                for call in calls:
                    assert call['function']['name']=='lookup'
                    assert json.loads(call['function']['arguments'])['key']=='alpha'
                    calls_seen+=1
                    history.append({'role':'tool','tool_call_id':call['id'],'content':'ERROR: temporary lookup failure. Retry alpha once.' if calls_seen==1 else large_result})
            else:
                r=responses(client,stream,model='gem16',input=history,tools=tools,reasoning={'effort':'none'},max_output_tokens=256)
                history.extend(r.output)
                calls=[item for item in r.output if item.type=='function_call']
                answer=r.output_text
                rounds.append(r.model_dump())
                for call in calls:
                    assert call.name=='lookup' and json.loads(call.arguments)['key']=='alpha'
                    calls_seen+=1
                    history.append({'type':'function_call_output','call_id':call.call_id,'output':'ERROR: temporary lookup failure. Retry alpha once.' if calls_seen==1 else large_result})
            if not calls:
                assert calls_seen==2 and marker in answer,(calls_seen,answer)
                return {'actual_calls':calls_seen,'result_bytes':len(large_result.encode()),'answer':answer,'rounds':rounds}
        raise AssertionError('tool error recovery exceeded four rounds')
    for api in ['chat','responses']:
        for stream in [False,True]:
            run(f'{api}/error-long-unicode/{stream}',lambda api=api,stream=stream:recover(api,stream))
    print(json.dumps({'sdk':openai.__version__,'cases':results,'passed':all(r['passed'] for r in results)},indent=2,ensure_ascii=False))
    return 0 if all(r['passed'] for r in results) else 1

if __name__=='__main__': sys.exit(main())
