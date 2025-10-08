#!/bin/bash

# set source and destination directories
SOURCE_DIR="magma"
DEST_DIR="external"

# ensure the source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
    echo "Source directory $SOURCE_DIR does not exist."
    exit 1
fi

# ensure the destination directory exists (create if it doesn't)
mkdir -p "$DEST_DIR"

# copy all files and directories from source to destination
cp -rf "$SOURCE_DIR/"* "$DEST_DIR/"

echo "Files from $SOURCE_DIR have been copied to $DEST_DIR and existing files have been overwritten."

# build custom magma container
cd $SOURCE_DIR || exit
tools/captain/shell_scripts/build_gcov_lua.sh

