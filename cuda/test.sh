#!/bin/sh

arg=$1

./sort $arg > before.txt

sort -n before.txt > after.txt

diff before.txt after.txt

rm before.txt after.txt
