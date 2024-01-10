#!/bin/bash

# Set the desired number of iterations
num_iterations=5
success_count=0
failure_count=0

# Loop the command
for ((i=1; i<=$num_iterations; i++)); do
    # Run the Python program
    rm -r ./.tmp
    rm -r ./output
    python main.py ./dataset/test-images-only -u
    
    # Check the exit status of the Python program
    if [ $? -eq 0 ]; then
        echo "Iteration $i: Success"
        ((success_count++))
    else
        echo "Iteration $i: Failure"
        ((failure_count++))
    fi
done

# Display results
echo "Finished with $success_count successful and $failure_count failed executions."
