#!/bin/bash

# Build project first
./build.sh

# Run tests
cd build && make test
