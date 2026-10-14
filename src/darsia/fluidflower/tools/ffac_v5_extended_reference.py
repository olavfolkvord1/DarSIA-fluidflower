"""Compatibility entry points for explicitly configured source-backed border strips."""
from __future__ import annotations
from darsia.fluidflower.native import sidecar as _sidecar
from darsia.fluidflower.native.config import require_parameters
from darsia.fluidflower.native.io import sha256_file as file_sha256

def __getattr__(name):
    if name=='CONTRACT':return require_parameters('extended_ring')['contract']
    raise AttributeError(name)

def split_extended_ring(padded_log,native_log,*,pad_px,maximum_center_delta_log=None):
    p=require_parameters('extended_ring')
    threshold=p['maximum_center_delta_log'] if maximum_center_delta_log is None else maximum_center_delta_log
    return _sidecar.split_extended_ring(padded_log,native_log,pad_px=pad_px,
        maximum_center_delta_log=threshold,parameters=p)
def write_extended_ring(path,padded_log,native_log,*,pad_px):
    return _sidecar.write_extended_ring(path,padded_log,native_log,pad_px=pad_px,parameters=require_parameters('extended_ring'))
def assemble_extended_log(native_log,ring_path,*,ring_sha256,expected_pad_px):
    return _sidecar.assemble_extended_log(native_log,ring_path,ring_sha256=ring_sha256,
        expected_pad_px=expected_pad_px,parameters=require_parameters('extended_ring'))

__all__=['CONTRACT','split_extended_ring','write_extended_ring','assemble_extended_log']
