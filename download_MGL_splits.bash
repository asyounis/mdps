#!/bin/bash

# Download the file
wget -O ./data/MGL_custom_splits/splits.zip https://www.dropbox.com/scl/fi/jwkzmtxbnxry70zblveww/splits.zip?rlkey=h487lyimtk6oayjtsfrhjdy6t&st=66o68b21&dl=0

# Unzip it
cd ./data/MGL_custom_splits/
unzip splits.zip