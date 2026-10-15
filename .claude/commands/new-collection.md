---
description: Create a collection from reference images, sweep t across [0, 1], build the contact sheet
argument-hint: <collection-name> <ref-image> [<ref-image> <ref-image>]
---

Create the genart collection `$1` from the reference images in: $ARGUMENTS
(the first word is the collection name, the rest are 1–3 image paths).

1. `uv run genart ingest <name> <images...>`
2. `uv run genart sweep <name> --n 24 --t-min 0 --t-max 1`
3. `uv run genart sheet <name>`

Then open `collections/<name>/sheet.html`, look at the renders, and critique them
against the aesthetic rules in CLAUDE.md. Report the genome ids worth mutating.
