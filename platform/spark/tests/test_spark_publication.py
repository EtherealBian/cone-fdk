"""Real Linux publication races, stale paths and source-pin controls (no GPU required)."""
import concurrent.futures
import hashlib
import os
import sys
import threading
from pathlib import Path

import pytest

HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
import build_publishers as bp
import spark_publication as sp


def test_two_real_publishers_keep_one_intact_result(tmp_path):
    dest=tmp_path/'result'; start=threading.Barrier(2)
    def contender(token):
        with sp.staging(dest,'.race-') as t:
            st=Path(t)/'completed';st.mkdir();(st/'payload').write_bytes(token)
            start.wait(5)
            try: sp.publish(st,dest)
            except FileExistsError: return None
            return token
    with concurrent.futures.ThreadPoolExecutor(2) as e:
        results=list(e.map(contender,[b'first',b'second']))
    kept=[r for r in results if r is not None]
    assert len(kept)==1 and (dest/'payload').read_bytes()==kept[0]
    assert set(tmp_path.iterdir())=={dest}


@pytest.mark.parametrize('kind',['empty-directory','file','dangling'])
def test_destination_introduced_during_staging_survives(tmp_path,kind):
    dest=tmp_path/'result'
    with sp.staging(dest,'.race-') as t:
        st=Path(t)/'completed';st.mkdir();(st/'payload').write_bytes(b'new')
        if kind=='empty-directory':dest.mkdir()
        elif kind=='file':dest.write_bytes(b'kept')
        else:dest.symlink_to('not-here')
        with pytest.raises(OSError):sp.publish(st,dest)
    if kind=='empty-directory':assert not list(dest.iterdir())
    elif kind=='file':assert dest.read_bytes()==b'kept'
    else:assert dest.readlink()==Path('not-here') and not (tmp_path/'not-here').exists()


def test_parent_swap_cannot_redirect_or_leak_output(tmp_path):
    true=tmp_path/'dir';true.mkdir();other=tmp_path/'other';other.mkdir();dest=true/'result'
    with sp.staging(dest,'.swap-') as t:
        st=Path(t)/'completed';st.mkdir();(st/'payload').write_text('new');true.rename(tmp_path/'old');true.symlink_to(other,True)
        with pytest.raises(ValueError,match='parent changed'):sp.publish(st,dest)
    assert not list(other.iterdir()) and not list((tmp_path/'old').iterdir())


@pytest.mark.parametrize('kind',['parent-link','file-link','dangling','dotdot'])
def test_raw_destination_checked_before_resolution(tmp_path,kind):
    real=tmp_path/'real';real.mkdir();link=tmp_path/'link'
    if kind=='parent-link':link.symlink_to(real,True);dest=link/'output'
    elif kind in ('file-link','dangling'):
        if kind=='file-link':(real/'new').write_text('unchanged')
        link.symlink_to(real/'new');dest=link
    else:dest=real/'..'/'output'
    with pytest.raises((ValueError,OSError)):sp.fresh(dest)
    assert not (real/'output').exists()


@pytest.mark.parametrize('name',sorted(bp.PINS))
def test_refuse_changed_frozen_source(name):
    with pytest.raises(ValueError,match='SHA-256'):bp.port(name,(bp.ROOT/'scripts'/(name+'.py')).read_bytes()+b'\n')


def test_reproducible_overlay_refuses_unexpected_cache(tmp_path):
    a,b=tmp_path/'a',tmp_path/'b'; assert bp.build(a)==bp.build(b)==bp.build(a)
    assert (a/'overlay.json').read_bytes()==(b/'overlay.json').read_bytes()
    (a/'unknown').write_text('keep')
    with pytest.raises(ValueError,match='not identical'):bp.build(a)
    assert (a/'unknown').read_text()=='keep'
