#!/bin/bash

mkdir dataset
mkdir dataset/train
mkdir dataset/test

for folder in Images/*
do  
    mkdir dataset/train/"$(basename $folder)"
    mkdir dataset/test/"$(basename $folder)"
done

i=0

for folder in Images/*
do
    i=0
    for image in "$folder"/*
    do
	if [ $i -lt 100 ]
	then
	    cp "$image" dataset/train/"$(basename $folder)"
	else
	    cp "$image" dataset/test/"$(basename $folder)"
	fi
	i=$((i+1))
    done
done


