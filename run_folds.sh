#!/bin/sh

MODEL=$1

echo "Compiling..."
mvn compile > /dev/null
mvn dependency:build-classpath -Dmdep.outputFile=classpath.out > /dev/null

echo "Running on fold 0..."
java -Xmx16g -cp ./target/classes:`cat classpath.out` edu.ucsc.cs.$MODEL data 0

echo "Running on fold 1..."
java -Xmx16g -cp ./target/classes:`cat classpath.out` edu.ucsc.cs.$MODEL data 1

echo "Running on fold 2..."
java -Xmx16g -cp ./target/classes:`cat classpath.out` edu.ucsc.cs.$MODEL data 2

echo "Running on fold 3..."
java -Xmx16g -cp ./target/classes:`cat classpath.out` edu.ucsc.cs.$MODEL data 3

echo "Running on fold 4..."
java -Xmx16g -cp ./target/classes:`cat classpath.out` edu.ucsc.cs.$MODEL data 4

echo "Running on fold 5..."
java -Xmx16g -cp ./target/classes:`cat classpath.out` edu.ucsc.cs.$MODEL data 5

echo "Running on fold 6É"
java -Xmx16g -cp ./target/classes:`cat classpath.out` edu.ucsc.cs.$MODEL data 6

echo "Running on fold 7..."
java -Xmx16g -cp ./target/classes:`cat classpath.out` edu.ucsc.cs.$MODEL data 7

echo "Running on fold 8..."
java -Xmx16g -cp ./target/classes:`cat classpath.out` edu.ucsc.cs.$MODEL data 8

echo "Running on fold 9..."
java -Xmx16g -cp ./target/classes:`cat classpath.out` edu.ucsc.cs.$MODEL data 9