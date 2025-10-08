#!/bin/bash
# change to coverage directory
cd $OUT/afl || exit

# execute lua interpreter on current testcase
./lua $SHARED/inputs/test.lua


