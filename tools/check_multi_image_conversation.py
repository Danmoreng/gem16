#!/usr/bin/env python3
"""Bounded live API regression: multiple images and image tool results in one slot.

Requires an explicitly started test server. Does not start/stop servers or open UI.
Uses only generated solid-color PNGs and writes an append-only evidence file.
"""
import argparse
import base64
import json
import re
import struct
import urllib.error
import urllib.request
import zlib
from pathlib import Path


def image_part(color):
    def chunk(kind, payload):
        return struct.pack('!I', len(payload)) + kind + payload + struct.pack('!I', zlib.crc32(kind + payload))
    png = b'\x89PNG\r\n\x1a\n'
    png += chunk(b'IHDR', struct.pack('!2I5B', 128, 128, 8, 2, 0, 0, 0))
    png += chunk(b'IDAT', zlib.compress((b'\0' + bytes(color) * 128) * 128))
    png += chunk(b'IEND', b'')
    return {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(png).decode()}}


def run(base_url, model, compact):
    messages = [{'role': 'system', 'content': 'Follow instructions precisely. Remember images in order. Use canvas_check only when explicitly asked.'}]
    tools = [{'type': 'function', 'function': {'name': 'canvas_check', 'description': 'Return new screenshots to this conversation.', 'parameters': {'type': 'object', 'properties': {}}}}]
    evidence = []
    session = None

    def request(label, new_message, use_tools=True):
        nonlocal session
        messages.append(new_message)
        body = {'model': model, 'messages': messages, 'stream': False, 'reasoning_effort': 'none', 'max_completion_tokens': 96}
        if use_tools:
            body['tools'] = tools
        if compact:
            body['vision_soft_token_budget'] = 70
        headers = {'Content-Type': 'application/json'}
        if session:
            headers['X-Gem16-Session-Id'] = session
        req = urllib.request.Request(base_url + '/v1/chat/completions', json.dumps(body).encode(), headers)
        try:
            with urllib.request.urlopen(req, timeout=180) as response:
                data = json.load(response)
                next_session = response.headers.get('X-Gem16-Session-Id')
        except urllib.error.HTTPError as error:
            raise RuntimeError(error.read().decode()) from error
        assert next_session and (session is None or next_session == session), 'resident session changed'
        usage = data['usage']
        cached = usage.get('prompt_tokens_details', {}).get('cached_tokens', 0)
        if session:
            assert cached > 0, 'continuation did not reuse cached tokens'
        session = next_session
        answer = data['choices'][0]['message']
        messages.append({key: value for key, value in answer.items() if key in ('role', 'content', 'tool_calls')})
        evidence.append({'step': label, 'session': session, 'usage': usage, 'answer': answer})
        print(label, json.dumps(evidence[-1]), flush=True)
        return answer

    def user(text, *images):
        return {'role': 'user', 'content': [{'type': 'text', 'text': text}, *images]}

    request('first_image', user('Remember this image. Name its color in one word. Do not call tools.', image_part((255, 0, 0))))
    request('append_image', user('Remember this second image. Name the colors of the first and second images, in that order. Do not call tools.', image_part((0, 0, 255))))
    call = request('request_screenshots', user('Call canvas_check now to obtain two more images.'))
    calls = call.get('tool_calls', [])
    assert len(calls) == 1 and calls[0]['function']['name'] == 'canvas_check', 'model did not request screenshots'
    request('append_two_tool_images', {'role': 'tool', 'tool_call_id': calls[0]['id'], 'content': [
        {'type': 'text', 'text': 'Two screenshots, images three and four. Describe the colors of all four images in chronological order. Do not call another tool.'},
        image_part((0, 255, 0)), image_part((255, 255, 0))]})
    recalled = request('recall_images', user('Name the colors of all four images in chronological order. Do not call tools.'))
    assert re.findall(r'\b(red|blue|green|yellow)\b', recalled.get('content', '').lower()) == ['red', 'blue', 'green', 'yellow'], 'historical image recall failed'
    # Explicit fresh root exercises reopening/replaying the full saved history.
    # Normal continuation above must preserve the resident session throughout.
    session = None
    replayed = request('fresh_history_replay', user('Again, name the colors of all four images in chronological order. Do not call tools.'))
    assert re.findall(r'\b(red|blue|green|yellow)\b', replayed.get('content', '').lower()) == ['red', 'blue', 'green', 'yellow'], 'image history replay failed'
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--model', default='gem16')
    parser.add_argument('--compact', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    # Reserve the output first; never overwrite prior evidence.
    with args.output.open('x') as output:
        evidence = run(args.base_url.rstrip('/'), args.model, args.compact)
        json.dump({'steps': evidence}, output, indent=2)
        output.write('\n')


if __name__ == '__main__':
    main()
