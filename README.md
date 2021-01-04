# Design Objectives

This is an attempt to create a universally (modern and older architecture)
performant Neural Network. 

The goals are:

1. *speed*. I wanna go fast. duh. Nuff said.

2. *memory*. 8 bit parameters (with 32 bit buffer) allow for a very low 
 memory overhead while retaining a healthy order of magnitude (255) for 
 representation expression. 
 It is possible to reduce this to 4 bit or even binary but this risks losing 
 expressivity and rewriting the architecture e.g: sigmoid requires some amount 
 of plasticity/representation/domain (for lack of a better term). 

3. *precision*. No loss from mantissa as found in FP networks that have 
 representation loss/redundancy that isn't really modelled with the matrix math
 (the first (-1, 1) sums have more weight than >1). This
 is not trivial with bitwise operations since squashing involves either 
 binning or many operations to create a function that meaningfully maps 32 to 
 8 bits. This should create consistent representation and 
 is in my current understanding a working flaw in Neural Networks without 
 "quantization".

4. *digitization*. This representation should be low level enough that a 
 few common logic gates/circuits should be able to reproduce or transfer 
 models into an FPGA or ASIC configuration with ease.

5. *energy effeciency* Low level operations should use less pipeline stages and therefore less silicon switching. 
 This network should be memory bottlenecked on all operations to the point parameter compression isnt off the 
 table with certain ALUs/architectures.
