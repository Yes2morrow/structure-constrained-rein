"""The same physical categories in exported figures and the editable canvas."""
from urllib.parse import quote

PARTITION_FILL = '#9ca3af'
DOOR_COLOR = '#facc15'
STRUCTURE_FILL = '#30343b'
CORE_FILL = '#fafafa'
CORE_EDGE = '#111111'
CORE_HATCH = '///'

# Fabric serializes and restores image patterns without changing object IDs.
_tile = '<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12"><rect width="12" height="12" fill="#fafafa"/><path d="M-3 3L3-3M0 12L12 0M9 15L15 9" stroke="#111111" stroke-width="1"/></svg>'
FABRIC_CORE_FILL = dict(type='pattern', source='data:image/svg+xml;charset=utf-8,'+quote(_tile), repeat='repeat')
