"""Integrate only the predeclared five-shape Assistant GEMV candidate."""
from pathlib import Path
import argparse
import hashlib
import json

ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('candidate_source',type=Path);p.add_argument('parent_sha256')
p.add_argument('receipt',type=Path);a=p.parse_args()
path=ROOT/'src/cuda/mtp/assistant.cu';source=path.read_text()
assert hashlib.sha256(path.read_bytes()).hexdigest()==a.parent_sha256
candidate=a.candidate_source.read_text()
begin=candidate.index('__global__ void Bf16GemvExactTreeCandidateKernel(')
end=candidate.index('\n}  // namespace',begin)
kernel=candidate[begin:end].replace('Bf16GemvExactTreeCandidateKernel','Bf16GemvExactTreeKernel')
assert kernel.count('__global__')==1 and kernel.count('__syncthreads()')==1
assert kernel.count('fmaf(')==2 and kernel.count('__shfl_down_sync')==1
marker='__global__ void SetAssistantAttentionPositionKernel('
assert source.count(marker)==1
source=source.replace(marker,kernel+'\n'+marker)
old='''    Bf16GemvKernel<<<static_cast<unsigned>(rows), kThreads, 0, stream>>>(
        input, weight, output, rows, contracting);'''
assert source.count(old)==1
new='''    const bool exact_tree_shape =
        (contracting == 1024U &&
         (rows == 4096U || rows == 8192U || rows == 3840U)) ||
        (rows == 1024U && (contracting == 4096U || contracting == 8192U));
    if (exact_tree_shape) {
      Bf16GemvExactTreeKernel<<<static_cast<unsigned>(rows), kThreads, 0, stream>>>(
          input, weight, output, rows, contracting);
    } else {
      Bf16GemvKernel<<<static_cast<unsigned>(rows), kThreads, 0, stream>>>(
          input, weight, output, rows, contracting);
    }'''
source=source.replace(old,new)
path.write_text(source)
with a.receipt.open('x') as f:
    json.dump({'owned_file':'src/cuda/mtp/assistant.cu','parent_sha256':a.parent_sha256,
               'candidate_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
               'scope':'only five declared12B Assistant GEMV shapes; PreProjection, head, other shapes and26B unchanged',
               'diagnostic_candidate_sha256':hashlib.sha256(a.candidate_source.read_bytes()).hexdigest()},f,indent=2)
