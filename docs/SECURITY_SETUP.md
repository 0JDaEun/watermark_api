# Repository hygiene

Local cache and secret patterns are ignored. The tracked root .DS_Store is removed in this review branch. Existing image samples may include identifiable people; confirm usage and publication rights before adding them to portfolio pages. Do not assume deletion from a branch removes files from Git history. No sample images were moved or republished.

## Tracked build artifacts

The review branch also removes the following generated paths from the Git tree:

- `app/__pycache__`
- `app/models/__pycache__`
- `app/utils/__pycache__`

Python recreates bytecode caches when required.
Historical Git objects remain unchanged. The application source, project data, and research artifacts are preserved.
