#!/bin/sh
# Usage: convert.sh IN_DIR OUT_DIR OUT_VERSION OUT_TYPE [FILTER]
#   e.g. convert.sh /work/in /work/dwg ACAD2018 DWG "*.dxf"   (then back: ... ACAD2007 DXF "*.dwg")
set -eu
BIN=$(command -v ODAFileConverter || ls /usr/bin/ODAFileConverter* /opt/ODAFileConverter*/ODAFileConverter 2>/dev/null | head -1)
mkdir -p "$2"
xvfb-run -a "$BIN" "$1" "$2" "$3" "$4" 0 1 "${5:-*.*}"
ls -la "$2"
for f in "$2"/*.err; do [ -f "$f" ] && { echo "== $f"; cat "$f"; }; done
exit 0
