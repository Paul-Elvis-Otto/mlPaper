#!/bin/bash

# Set the URL (replace with your actual URL)
URL="https://v-dem.net/media/datasets/V-Dem-CY-FullOthers-v15_csv.zip"

# Create a temporary file for the download
TMP_FILE=$(mktemp)

# Create the target directory if it doesn't exist
mkdir -p vdemData

# Download the file
echo "Downloading from $URL..."
curl -L -o "$TMP_FILE" "$URL"

# Check if download was successful
if [ $? -ne 0 ]; then
  echo "Error: Download failed"
  rm "$TMP_FILE"
  exit 1
fi

# Unzip the file to the target directory
echo "Extracting files to vdemData directory..."
unzip -q -o "$TMP_FILE" -d vdemData

# Check if unzip was successful
if [ $? -ne 0 ]; then
  echo "Error: Extraction failed"
  rm "$TMP_FILE"
  exit 1
fi

# Clean up the temporary file
rm "$TMP_FILE"

echo "Success! Files have been downloaded and extracted to vdemData directory."
