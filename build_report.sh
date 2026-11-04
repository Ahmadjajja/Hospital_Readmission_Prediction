#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python run_experiments.py
python summarize.py
python verify_results.py
python generate_report_results.py
cd paper
pdflatex -interaction=nonstopmode -halt-on-error report.tex
bibtex report
pdflatex -interaction=nonstopmode -halt-on-error report.tex
pdflatex -interaction=nonstopmode -halt-on-error report.tex
