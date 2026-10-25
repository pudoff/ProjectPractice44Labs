#!/bin/sh
# Usage: validate.sh /work/file.dxf [paper WxH in mm, default 420x297]
# Writes next to the input: <name>.pdf (LibreCAD print), <name>.librecad.png (150 dpi preview),
# <name>.librecad.log. Exit 1 when LibreCAD cannot load/print the file.
set -u
f="$1"
paper="${2:-420x297}"
dir=$(dirname "$f")
base=$(basename "$f" .dxf)

echo "== LibreCAD dxf2pdf ($paper mm, scale 1:1)"
rm -f "$dir/$base.pdf"
timeout 120 librecad dxf2pdf -a -p "$paper" -r 300 -t "$dir" "$f" >"$dir/$base.librecad.log" 2>&1
if [ -s "$dir/$base.pdf" ]; then
  echo "OK  $base.pdf ($(stat -c %s "$dir/$base.pdf") bytes)"
  pdftoppm -png -r 150 -singlefile "$dir/$base.pdf" "$dir/$base.librecad" && echo "OK  $base.librecad.png"
  pdffonts "$dir/$base.pdf" 2>/dev/null | tail -n +3 | awk '{print "font:", $1}' | sort -u
else
  echo "FAIL librecad could not print (see $base.librecad.log)"; exit 1
fi
