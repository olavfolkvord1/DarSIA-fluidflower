"""Audited cache identities for changes that only move diagnostic work.

The listed before/after adapters return the same optical float32 heads. Their
CPU comparisons and report labels differ; model, image, feature, radiometry,
hardware and library identities are still compared independently. Unknown
source revisions keep their actual digest and cannot inherit these entries.
"""

AUDITED_STORAGE_IMPLEMENTATIONS = {
    'darsia.fluidflower.phase_training.gas_negative_sampling': (
        '07a390c5c931089732b6c01ed4a493f7499a5b743e19e4d58729a14f0f705678',
        {'07a390c5c931089732b6c01ed4a493f7499a5b743e19e4d58729a14f0f705678',
         'b609b8fa6670305504b20ff2dc515e6dc3710598fe505e21704c6a05c3cf8c0e'},
    ),
    'scripts.phase_rig_registration': ('b544ae68ddd320e0e3a9e4e0beca9e59843f5731a5d03078c3df6bfeca4b3f89', {'b544ae68ddd320e0e3a9e4e0beca9e59843f5731a5d03078c3df6bfeca4b3f89', '30d84e609118ceef466be3ee37bd1aa47aa3c314d03d0f54c866af8a9b8cf18c'}),
    'darsia.fluidflower.phase_training.radiometry': ('ea769f743c892bb2a299b44c9879f32c88d94e1758130692b43cf9ffbd890798', {'ea769f743c892bb2a299b44c9879f32c88d94e1758130692b43cf9ffbd890798', '81954f41fbde2b541d387fe92ca52ed3df8cb96a6a6775807a40808fa0b955f8'}),
}


AUDITED_IMPLEMENTATIONS = {
    'cpu_parity_reference': (
        '06ab6ede87566437586e298e545e2ab7f20f0f10d44e9676286830b6bda4b215',
        {'06ab6ede87566437586e298e545e2ab7f20f0f10d44e9676286830b6bda4b215', '6267c42aa0124f1830510034b8f75918e4dd12eaaf0c9a8d9d4d5c92bd676238'},
    ),
    'CheckedCuda': (
        '6f9965e6457e0027d4b81159168a3c08f5438045a63d8765a09aa1f174a60a7e',
        {'6f9965e6457e0027d4b81159168a3c08f5438045a63d8765a09aa1f174a60a7e',
         'c7bb4707e4472f7b72584063b356b6446367724128b4e4e5cb18431a386def25',
         'bee9b618133bb92d8ef88e460d21526a2b1ec11653f0b59136afd8f15df09ec7'},
    ),
    'CpuPredictor': (
        '0f54dea72bc69c3c273e863ac134c628f9a4959a4015d823cc5c158cf85d5e83',
        {'0f54dea72bc69c3c273e863ac134c628f9a4959a4015d823cc5c158cf85d5e83',
         '0cdd2d292a43238625ef010e0c5831d411841c4e7e7c5daceac946c489271e30',
         '985ca3a5d40870502ccabea9a3b90f6abdb16cba9d4ef47cf9bcf23da75a834e'},
    ),
    'scripts.phase_optical_cache': (
        'a3aab686a73c288110596eafd2fc1ffa1567db41e7855913fdbf3efdebf6e8fb',
        {'a3aab686a73c288110596eafd2fc1ffa1567db41e7855913fdbf3efdebf6e8fb',
         '296c3f52e63d6679ee00c01097de9eae621d09dcec07999eea6dc4b23a4fc4a6',
         'e53ca9c108d654f8ead4be32e9a31494c596c0398842583f3d386f65b8f3d3ac',
         # A bounded usage ledger replaces repeated directory scans. Payload
         # creation, hashes, validation and numerical optical heads are unchanged.
         'e5a7d3683cc68ab12ea4403b81e1a24667f9a0174944fba981aa0b11c21491bc'},
    ),
}


def canonical_implementation(name, digest):
    key = canonical_implementation_name(name)
    known = AUDITED_IMPLEMENTATIONS.get(key) or AUDITED_STORAGE_IMPLEMENTATIONS.get(key)
    return known[0] if known is not None and digest in known[1] else digest


def canonical_implementation_name(name):
    """Retain the recorded identity of a tool moved into the installed package."""
    prefix = 'darsia.fluidflower.tools.'
    return 'scripts.' + name[len(prefix):] if name.startswith(prefix) else name
