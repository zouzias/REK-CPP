#!/usr/bin/env bash

set -o pipefail

echo "***************************"
echo "***************************"
echo "*      Valgrind           *"
echo "***************************"
echo "***************************"
pushd build
valgrind --leak-check=full --track-fds=yes --track-origins=yes --leak-check=full --show-leak-kinds=all --error-exitcode=1 ./build/bin/test_sparse