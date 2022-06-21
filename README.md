# CS 5542 Final Project: Bitonic Sort

Authors: Jacob Villemagne and Curt Bridgers

## Bitonic Sort

[TODO: Write the rest of this]

## Time Sheets

This section contains a log of the amount of time spent working on this
project. Each author has a subsection in which their time was logged.

### Curt Bridgers

June 15, 2022: 9am-11am (Got the sequential bitonic sort working)

June 15, 2022: 2pm-4pm (Class Time: Started cuda implementation)

June 16, 2022: 2pm-4pm (Class Time: Thinking...)

June 17, 2022: 10am-11am (Setup the cuda code with makefiles, helpers, etc.
Can't get any print outs from the kernels...)

June 17, 2022: 11:30am-12:30pm (Got the indexing math working during the initial
kernel calls for each stage of the algorithm. The values retrieved from arr
are wrong, though. Probably some issue with the copy from host to device.)

June 17, 2022: 4pm-5pm (Fixed the weird memory issue. Added the memcpy from
device to host after the sort is complete. Added swap code for the top level
of each stage. Need to figure out how I want to handle the intermediary swapping
steps of each stage. Another loop that calls more kernels? Recursive kernel
launches? Decisions decisions...)

June 20, 2022: 2pm-4pm (Added a convenient way of toggling debug
output/behavior. Need to add an inner loop that launches kernels for each 
intermediary step for the stages. Made the algorithm use more than a single
CUDA block allowing us to sort meaningfully large arrays. I.E. Arrays of size
greater than 2^21. I think this should be completely working by tomorrow.)

June 20, 2022: 9pm-10pm (Colorized output to reduce eye strain trying to parse
it when debugging. Quality of life improvement that I should have done ages
ago...)

### Jacob Villemagne

June 15, 2022: 2pm-4pm (Class Time: Started OpenMPI implementation)
