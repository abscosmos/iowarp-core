# Quick Header Guide

## TL;DR

**Update existing files:**
```bash
./update_headers.sh --dry-run    # Preview changes
./update_headers.sh              # Apply changes
```

**New files in VSCode:**
1. Install "psioniq File Header" extension
2. Create new file
3. Press `Ctrl+Alt+H` (or `Cmd+Alt+H` on Mac)

## Common Commands

```bash
# Preview what would change (safe, no modifications)
./update_headers.sh --dry-run

# Update all files in repo
./update_headers.sh

# Update specific directory
./update_headers.sh --path context-transfer-engine/

# Update single file
./update_headers.sh --file src/myfile.cc

# Get help
./update_headers.sh --help
```

## VSCode Extension Setup

1. Open VSCode
2. Extensions (Ctrl+Shift+X)
3. Search "psioniq File Header"
4. Click Install
5. Reload window

Settings are already configured in `.vscode/settings.json`.

## Keybinding

- **Insert Header**: `Ctrl+Alt+H` (Windows/Linux) or `Cmd+Alt+H` (Mac)
- **Or**: Command Palette → "Insert Header"

## The Header

```c
/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 * [... BSD 3-Clause License ...]
 */
```

Full header text is in `.header_template`.

## Files Created

- `update_headers.py` - Python script to update headers
- `update_headers.sh` - Bash wrapper for convenience
- `.header_template` - Header template file
- `.vscode/settings.json` - VSCode configuration (updated)
- `.vscode/extensions.json` - Recommended extensions (created)
- `HEADER_UPDATE_GUIDE.md` - Full documentation
- `QUICK_HEADER_GUIDE.md` - This file

## Need Help?

See `HEADER_UPDATE_GUIDE.md` for complete documentation.
