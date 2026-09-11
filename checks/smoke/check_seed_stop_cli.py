"""Real process: stop during an episode, then resume that episode and decode."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime

ROOT=Path(__file__).resolve().parents[2]


def main():
    out=ROOT/'results2/seed_experimental'/('cp5_stop_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    flag=out.parent/(out.name+'.stop')
    log=out.parent/(out.name+'.log')
    env=dict(os.environ,PYTHONIOENCODING='utf-8')
    command=[sys.executable,'-u',str(ROOT/'scripts/train_seed_layout.py'),'--episodes','4','--max-steps','64',
             '--output',str(out),'--stop-file',str(flag)]
    with log.open('w',encoding='utf-8') as stream:
        process=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,env=env,
                                 creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        try:
            deadline=time.monotonic()+60
            while time.monotonic()<deadline:
                path=out/'latest_snapshot.json'
                if path.exists():
                    snapshot=json.loads(path.read_text(encoding='utf-8'))
                    if snapshot['step']>=10:
                        flag.write_text(f'pid={process.pid}\n',encoding='utf-8')
                        break
                if process.poll() is not None: raise AssertionError(log.read_text(encoding='utf-8'))
                time.sleep(.05)
            else: raise AssertionError('未观察到回合内持久快照')
            if process.wait(timeout=60)!=0: raise AssertionError(log.read_text(encoding='utf-8'))
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=30)
    stopped=json.loads((out/'run_status.json').read_text(encoding='utf-8'))
    assert stopped['status']=='stop_requested',stopped
    assert stopped['snapshot']['completed_episodes']==0 and 0<stopped['snapshot']['step']<64,stopped
    resume=[sys.executable,str(ROOT/'scripts/train_seed_layout.py'),'--resume',str(out),'--episodes','1','--stop-file',str(flag)]
    with log.open('a',encoding='utf-8') as stream:
        subprocess.run(resume,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=60)
    finished=json.loads((out/'run_status.json').read_text(encoding='utf-8'))
    assert finished['status']=='finished' and finished['snapshot']['completed_episodes']==1,finished
    assert len((out/'estimated_metrics.jsonl').read_text(encoding='utf-8').splitlines())==1
    result=dict(run_dir=str(out),stop=stopped,resume=finished,
                precise_outputs=len(list((out/'precise').glob('*/result.json'))))
    (ROOT/'checks/evidence/seed_stop_cli_result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
