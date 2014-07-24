#!/bin/sh

MODEL=$1
FOLD=$2

echo "Compiling..."
mvn compile > /dev/null
mvn dependency:build-classpath -Dmdep.outputFile=classpath.out > /dev/null

echo "Running on fold 0..."
java -Xmx16g -cp ./target/classes:`cat classpath.out` edu.ucsc.cs.$MODEL data $FOLD
