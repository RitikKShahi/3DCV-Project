#!/usr/bin/env bash
# Compile script for IEEE Conference Project Report
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

echo "Compiling IEEE Conference Report (report.tex)..."

if command -v pdflatex &> /dev/null; then
    pdflatex -interaction=nonstopmode report.tex
    pdflatex -interaction=nonstopmode report.tex
    echo "✓ Successfully generated report.pdf!"
elif command -v tectonic &> /dev/null; then
    tectonic report.tex
    echo "✓ Successfully generated report.pdf with Tectonic!"
else
    echo "Notice: Neither pdflatex nor tectonic was detected on your PATH."
    echo "To compile report.pdf locally, install BasicTeX or MacTeX:"
    echo "   brew install --cask basictex"
    echo "Alternatively, upload report.tex and IEEEtran.cls directly to Overleaf (https://www.overleaf.com) to download report.pdf instantly."
fi
