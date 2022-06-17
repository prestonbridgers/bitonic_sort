# Bitonic Sort OpenMPI Implementation
Jacob Villemagne's Work here

--Jun 15 Class hours--
2:00 - 4:00 Working on getting source code to work

-- Jun 16 Out of class hours --
12:00 - 2:00 Working on my Bitonic sorting code
    -- Copying given code on bitonic sorting
    -- Getting that code to compile
    -- Started working on the main method needed to run all of the methods
-- Jun 16 Class hours --
2:00 - 4:00 Working on my Bitonic sorting code
    -- Initalising MPI
    -- Getting numP and myId
    -- Process 0 prompting the user for the size of the dataSet we want to sort
-- Jun 17 Out of class hours --
11:00 - ...
    -- Decided to scrap the idea of process 0 promping the user for a input value
    -- Switched to using a command line argument
    -- Now all threads are receving the size variable
    -- Process 0 is creating the dataSet
        -- dataSet array is dyamically allocated for 2^(user input)
        -- Process 0 filling the array with numbers
        -- Randomizing the array so we can sort the numbers
        -- All process using the size varable to calculate how large the sliceArray needs to be
        -- The sliceArray will be the chunch of data each array works on sorting
    -- Currently working on process 0 sending all of the other processes there chunck of data
     