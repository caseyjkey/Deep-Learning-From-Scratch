#!/bin/bash

# Script to generate submission PDF for any assignment folder
# Usage: ./make_submission_pdf.sh [folder_name]
# If no folder name provided, defaults to "Co-Occurence Coding"

set -e

# Get folder name from parameter or use default
FOLDER_NAME="${1:-Co-Occurence Coding}"

if [ ! -d "$FOLDER_NAME" ]; then
    echo "Error: Assignment folder '$FOLDER_NAME' not found"
    echo "Available folders:"
    ls -d */ | grep -v ".*\.git.*" | sed 's|/||g'
    exit 1
fi

cd "$FOLDER_NAME"

echo "Generating submission.pdf for '$FOLDER_NAME'..."

# Check if submission.tex exists
if [ ! -f "tex/submission.tex" ]; then
    echo "Error: tex/submission.tex not found"
    exit 1
fi

# Generate PDF using direct pdflatex command
cd tex
env -u LATEXMKRC pdflatex -interaction=nonstopmode submission.tex > /dev/null 2>&1

if [ -f "submission.pdf" ]; then
    echo "✓ PDF generated successfully: submission.pdf"
    echo "  Assignment: $FOLDER_NAME"
    echo "  Location: $(pwd)/submission.pdf"
    echo "  Size: $(du -h submission.pdf | cut -f1)"
    echo ""
    echo "Note: The answers from solution.txt are already integrated into submission.tex."
else
    echo "✗ Error: PDF generation failed. Check submission.log for details."
    exit 1
fi
