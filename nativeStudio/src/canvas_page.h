#pragma once
#include "canvas.h"
#include "media_loader.h"
namespace gem16::studio {
// The generated document lives in an opaque-origin iframe. The outer document
// has no model code and collects bounded observations, never privileged
// commands.
inline std::string CanvasPage(const CanvasDocument& d) {
  const auto& source = d.revisions.back().source;
  const std::string boot = R"(<script>
(()=>{const report=s=>parent.postMessage({canvasDiagnostic:String(s).slice(0,2000)},'*');
addEventListener('error',e=>report(e.message||'Resource failed'),true);
addEventListener('unhandledrejection',e=>report('Promise: '+e.reason));
addEventListener('securitypolicyviolation',e=>report('Blocked resource: '+e.blockedURI));
for(const level of ['warn','error']){const old=console[level];console[level]=(...args)=>{report(level+': '+args.join(' '));old.apply(console,args)}};
})();</script>)";
  const std::string prefix =
      R"(<!doctype html><meta charset="utf-8"><meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:; font-src data:; connect-src 'none'; frame-src 'none'; base-uri 'none'; form-action 'none'; object-src 'none'">)" +
      boot;
  const auto child = prefix + source;
  const auto encoded =
      EncodeBase64(std::vector<std::uint8_t>(child.begin(), child.end()));
  return R"(<!doctype html><meta charset="utf-8"><meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; frame-src about: 'self'; connect-src 'none'; base-uri 'none'; form-action 'none'"><style>html,body,iframe{margin:0;width:100%;height:100%;border:0;background:white}html,body{overflow:hidden}iframe{display:block;position:absolute;inset:0}</style><iframe sandbox="allow-scripts"></iframe><script>
window.__canvasDiagnostics=[];window.__canvasReady=false;
const frame=document.querySelector('iframe');
function report(s){if(__canvasDiagnostics.length<12)__canvasDiagnostics.push(String(s).slice(0,2000))}
addEventListener('message',e=>{if(e.source===frame.contentWindow&&e.data&&typeof e.data.canvasDiagnostic==='string')report(e.data.canvasDiagnostic)});
frame.onload=()=>{window.__canvasReady=true};
const decode=s=>new TextDecoder().decode(Uint8Array.from(atob(s),c=>c.charCodeAt(0)));
)" "const child=decode('" +
         encoded + "');" +
         (d.type == "svg"
              ? "const parsed=new DOMParser().parseFromString(child.slice(" +
                    std::to_string(prefix.size()) +
                    "),'image/svg+xml');const "
                    "error=parsed.querySelector('parsererror');if(error)report("
                    "'SVG parse error: '+error.textContent);"
              : "") +
         "frame.srcdoc=child;</script>";
}
inline constexpr const char* kCanvasObservationScript =
    "JSON.stringify({ready:window.__canvasReady===true,diagnostics:(window.__"
    "canvasDiagnostics||[]).join('\\n')})";
}  // namespace gem16::studio
