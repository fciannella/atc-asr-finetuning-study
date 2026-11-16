from pathlib import Path
import hashlib,json,subprocess,torch,nemo,lightning,lhotse
R=Path(__file__).parent
N=Path('/opt/prompt_model/NeMo')
x={'torch':torch.__version__,'nemo':nemo.__version__,'lightning':lightning.__version__,'lhotse':lhotse.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'nemo_commit':subprocess.check_output(['git','-C',str(N),'rev-parse','HEAD'],text=True).strip(),'source_sha256':{str(p.relative_to(N)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [N/'examples/asr/speech_to_text_finetune.py',N/'examples/asr/speech_to_text_eval.py',N/'nemo/collections/common/data/lhotse/dataloader.py']}}
(R/'runtime.json').write_text(json.dumps(x,indent=2)+'\n');print('RUNTIME',json.dumps(x),flush=True)
