#!/bin/bash

# Function to compile a single tex file
compile_tex() {
    local tex_file="$1"
    local base_name="${tex_file%.tex}"

    echo "=========================================="
    echo "Compiling: $tex_file"
    echo "=========================================="

    # 1. First pass with xelatex
    echo "Pass 1: xelatex..."
    xelatex -interaction=nonstopmode "$tex_file" > /dev/null

    # 2. Run bibtex if a .bib file exists or if citations are found
    # We check for the .aux file to run bibtex on
    if grep -q "bibdata" "$base_name.aux"; then
        echo "Running bibtex..."
        bibtex "$base_name" > /dev/null
    fi

    # 3. Second pass with xelatex (for references/citations)
    echo "Pass 2: xelatex..."
    xelatex -interaction=nonstopmode "$tex_file" > /dev/null

    # 4. Third pass with xelatex (for proper resolving of cross-references)
    echo "Pass 3: xelatex..."
    xelatex -interaction=nonstopmode "$tex_file" > /dev/null

    echo "Done with $tex_file"
    echo ""
}

# Loop through all .tex files in the current directory
# We explicitly list them or just find *.tex. 
# Given the directory structure, *.tex in the root is safe.
for f in *.tex; do
    if [ -f "$f" ]; then
        compile_tex "$f"
    fi
done

echo "All compilations finished."
