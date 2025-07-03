#!/bin/bash
# This script is used to push changes to the GitHub repository
git add .
git commit -m "Update files"
git push origin main
# Check if the push was successful
if [ $? -eq 0 ]; then
    echo "Changes pushed to GitHub successfully."
else
    echo "Failed to push changes to GitHub."
fi
