// READ THIS AT THE PERIL OF MY SPARSLEY RECORDED INNER MACHINATIONS //
//
//
// NOTE: NORMALIZATION IS THE KEY TO FAST AND LIGHTWEIGHT NEURAL NETWORKS
// NOTE: the pipeline has been created with cond_rot_act for how to rotationally approximate equations: prescribe a 
//       function then trace with offsets and rotations with piecewise conditions. If multiplication is a 
//       saddle function what features are necessary for expressivity/approximation that can be done with rot ops?
//          1. f'''(x)/dx !== 0 (concavity and inflection).
//          2. symmetry about said inflection.
//       with 1 and 2 regularity is found wrt sigmoid activation function and multiplication function. pretty interesting!
//
//       rot ops hold as equivalent and therefor optimal given the hypothesis that the connection of the graph holds 
//       information and is only expressed with these basic terms of expression. This can be found in various applications
//       of information theory such as compiler optimization and the shannon quantification of information 
//       (in higher dimension than the paper described e.g.:tree).
//
// TODO: rot connection weight repeats due to bit shift since 0 overflow shift implemented.
//       set max weight value to index precision or change otherwise.
//       (scrubbing (notch_rot?) has combination of data type length distinct
//       outputs f(x) != {f({y})} | x !e y: len(dtype)!)
//       can conditional inequality piecewise scrub. This yields a smooth function
//       that can be conditionally shaped to saddle points. at what condition+binop count
//       is mult faster (at scale, considering SIMD etc)? is this faster than unsigned integer
//       multiplication because thats what is is unless creating a function that isnt a saddle.
//  Can also prescribe a bitop approximated function just as with cond_rot_act. multiplication is just a saddle in R3
//  when A*B is swept for A and B s.t. Z = f(A,B)
// TODO: notch_rot is more expressive if can be faster than multiplication and 
//       can be piecewise differentiable. conditions are good because although they are not SIMD they 
//       are branch predicted and <=1 cycle (jump and compare is in every ISA). 
//       Cannot be derived on shift with zero padding architectures without data-structure bloat.
//
// NOTE: if this can be differentiated to a novel non genetic algorithm architecture search 
//       name it Psyclone as an homage to my birthplace and due to the thorough use of 
//       rotations and the mersenne twister PRNG.
//
// TODO: 16 conditions for non isotropic function(continually increasing/decreasing).
//       mirror 8 conditions for a saddle or smoother quadratic curve (convex)
// TODO: notch rot is possibly better if not using gradient based optimization/AS. Write this as a method to connections in 
//       parallel with multiplication. 
// TODO: its possible to take the gradient of these functions, adagrad may make this trivial. worst case need to store 
//       signal through forward prop for back prop but this is just a byte
//
// NOTE: for now, multiplication is plenty fast and well optimized. 
//      Using rotation since the oscillations in the domain are a feature. 2^n -modal 


// Can connections be reused outside of networks to allow address innovation? using rust Arc.clone/copy this may be possible 
// but may have some parallel overhead. would reduce the genepool-parameter overhead dramatically.
// all identifiers should be address based.
// TODO: REALLY NEED ADDRESSABLE INNOVATION

// NOTE: look at Polly for llvm and BLAS for parallelization/SIMD optimization. prefer to get into matrix ops and call on GPU

// NOTE: Vecs are chunks of locality that is why capacitance exists due to the total re allocation on capacity exceeding.
//       Loading a vec on the stack loads sequential addresses.
// NOTE: on the cache locality of graphs: 
//      Premise: using let Node.in/out_connections = Box<[Connections; N]>:
//
//      1. The topology search can be constrained to a maximum amount of parameters given size of memory resources 
//         (nodes+connections) == memory
//
//      2. The topology search can also be constrained by the number of connections per node to get optimal 
//         cache locality in forward prop/back prop etc. (out_connections || in_connections) == (L1/shared L2 cache)/(num_threads).
//         since Box<[]> dereferences cache locality wrt the stack and heap.
// TODO: just use chunk iterator for now. slice a vec using to_slice method to a known array chunk size. 
//       (can this be static const configured as a build param?
/////////// SHADOW ARRAYS ///////////
// rust is currently data structure deficient. Not trivial due to type safety.
// compare this to unsized locals for safety. should have a Vec method for casting into 
// Shadow Array for safely wrapped unsafe dyn sized stack array just to keep this feature organized.
// TODO: Shadow Arrays <[]>. a Vector but with tuple instead of triple. allocation occurs at every mutation and 
//       entries are shadowed in. fast access slow allocated mutation. (ptr, len). dereferencing bring an entire array onto the stack.
//       shadowed for safety entries live on heap behind a Cell. essentially: Cell<Box<[unsized-unsafe]>> where cell is 
//       changed between boxes during shadowing. cell: [1,2] shadowed_new_cell: [1,2,3] cell.set(shadowed_new_cell)
//      serves 4 purposes: 
//      1. lighterweight at scale, removes 1 usize from all would-be vec
//      2. memory efficiency, never over allocates. (with worse allocation delays)
//          used for lots of reads/dereferencing with sparse mutation.
//      3. faster dereferencing since not striding over pointers on the heap, the entire array gets dereferenced
//         1 layer of indirection. (going to be pretty damn unsafe)
//      4. safely wrap dynamicly sized stack arrays with a standard implementation.
//      HOWTO: https://doc.rust-lang.org/nomicon/vec.html
//      TODO: warn about stack overflow since cell is essentially shadowing with copy. 
//            lazy iterators that push scope for move may make this easier.
/////////////////////////////////////
