"""One finite, event-driven source-to-qualification job. No retry, polling loop or local fallback."""
import argparse
import fcntl
import os
import select
import subprocess
from datetime import datetime
from pathlib import Path

HERE=Path(__file__).resolve().parent
from prepare_pine import ROOT, pine


def wait_for_acquisition(unit, invocation, receipt, timeout):
    def state():
        r=subprocess.check_output(['systemctl','--user','show',unit,'--property=MainPID,InvocationID'],text=True)
        return dict(row.split('=',1) for row in r.splitlines())
    if receipt.is_file(): return
    info=state();pine.need(info['InvocationID']==invocation and int(info['MainPID'])>0,'acquisition is not the named running instance')
    fd=os.pidfd_open(int(info['MainPID']))
    try:
        pine.need(state()==info or receipt.is_file(), 'acquisition replaced while binding')
        if not receipt.is_file():
            poll=select.poll();poll.register(fd,select.POLLIN)
            pine.need(bool(poll.poll(max(0,int(timeout*1000)))), 'source deadline expired')
    finally:os.close(fd)
    pine.need(receipt.is_file(),'acquisition ended without a valid completion receipt; no reconstruction attempted')


def main(args):
    import prepare_pine, qualify_pine
    deadline=datetime.fromisoformat(args.deadline);pine.need(deadline.tzinfo is not None,'timezone required')
    left=lambda: deadline.timestamp()-datetime.now().timestamp()
    pine.need(left()>0,'window closed');M=ROOT.parent; originals=ROOT/'data/pine-cone-originals';receipt=originals/'source-receipt.json'
    stage=args.stage.absolute();result=args.output.absolute();pine.need(stage.parent==result.parent==M and not stage.exists() and not result.exists() and stage!=result,'new migration-local job paths required')
    print('Waiting on the acquisition process; final receipt required.',flush=True)
    wait_for_acquisition(args.unit,args.invocation,receipt,left());pine.need(left()>0,'window closed before preprocessing')
    print('Acquisition complete; revalidating all archives and reconstructing independent subsets.',flush=True)
    prepare_pine.run(originals,stage);pin=pine.sha(stage/'scan.json')
    print('Accepted preprocessing receipt:',pin,flush=True)
    with (M/'gpu.lock').open('rb') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);pine.need(left()>0,'window closed before GPU work')
        qualify_pine.run(stage,receipt,ROOT/'build/spark',result,pin)
    print('Qualification receipt:',pine.sha(result/'qualification.json'),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('unit','invocation','deadline'):p.add_argument(n)
    for n in ('stage','output'):p.add_argument(n,type=Path)
    main(p.parse_args())
