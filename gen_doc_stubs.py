# gen_doc_stubs.py
from pathlib import Path

import mkdocs_gen_files

this_dir = Path(__file__).parent

# Pull hand-written design specs (outside docs_dir) into the docs tree so
# they can be referenced from SUMMARY.md like any other page.
specs_root = (this_dir / "../specs").resolve()
for spec_name in ("hierarchy.md",):
    spec_path = specs_root / spec_name
    with mkdocs_gen_files.open(spec_name, "w") as fd:
        fd.write(spec_path.read_text())
    mkdocs_gen_files.set_edit_path(spec_name, Path("../specs") / spec_name)

src_root = (this_dir / "../circulax").resolve()
excluded_api_paths = {Path("components") / ("va_" + "component.py")}

for path in src_root.rglob("*.py"):
    rel_path = path.relative_to(src_root)
    if rel_path.parts[0] == "va" or rel_path in excluded_api_paths:
        continue

    parts = tuple(rel_path.with_suffix("").parts)

    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = rel_path.with_name("index.md")
    elif parts[-1] == "__main__":
        continue
    else:
        doc_path = rel_path.with_suffix(".md")

    full_parts = ("circulax",) + parts
    identifier = ".".join(full_parts)

    output_filename = Path("references") / doc_path

    with mkdocs_gen_files.open(output_filename, "w") as fd:
        fd.write(f"::: {identifier}")

    mkdocs_gen_files.set_edit_path(output_filename, path)
