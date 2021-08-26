# Design Objectives

This is an attempt to create a universally (modern and older architecture)
performant Neural Network. 

The goals are:

1. *speed*. I wanna go fast. duh. Nuff said.

2. *memory*. 8 bit parameters allow for a very low 
 memory overhead while retaining a healthy order of magnitude (255) for 
 representation expression. 
 It is possible to reduce this to 4 bit or even binary but this risks losing 
 expressivity and rewriting the architecture e.g: sigmoid requires some amount 
 of plasticity/representation/domain (for lack of a better term-- it would not be as expresive).
 this also should be limited by the smallest word size for the architecture which is generally x8 bits.

3. *precision*. No loss from mantissa as found in floating point network operations that have 
 representation loss/redundancy that isn't really modelled with the matrix math
 (the first (-1, 1) sums have more precision than >1 which leads to bias, strange normalization 
 techniques etc).

4. *digitization*. This representation should be low level enough that a 
 few common logic gates/circuits should be able to reproduce or transfer 
 models into an FPGA or ASIC configuration with ease.

5. *energy effeciency*. Low level operations should use less pipeline stages and therefore less silicon switching. 
 This network should be memory bottlenecked on all operations to the point parameter compression isnt off the 
 table with certain ALUs/architectures that have pipelines that are partially idling. while it is possible to use binary and
 scale out connections, this representation aims to be a good solution for both and therefor a more universal model.


config contains the rustc and llvm params for this build.
