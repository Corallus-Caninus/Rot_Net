// TODO: attempt normalized connections. after sum normalize prior to activation
//       (squash then activate-- activation is for network emergent approximation not squashing)
//       this would mean the entire network is u8 and wont need 32 buffer extension which doesnt
//       carry representation anyways without a larger activation domain.
//
// TODO: rot connection weight repeats due to bit shift since 0 overflow shift implemented.
//       set max weight value to index precision or change otherwise.
//       (scrubbing (notch_rot?) has combination of data type length distinct
//       outputs f(x) != {f({y})} | x !e y: len(dtype)!)
//       can conditional inequality piecewise scrubbing. This yields a smooth function
//       that can be conditionally shaped to saddle points. at what condition+binop count
//       is mult faster (at scale, considering SMD etc)? is this faster than unsigned integer
//       multiplication because thats what is is unless creating a function that isnt a saddle.
// TODO: notch_rot is more expressive if can be faster than multiplication and 
//       can be piecewise differentiable.
//
// TODO: 16 conditions for non isotropic function(continually increasing/decreasing).
//       mirror 8 conditions for a saddle or smoother quadratic curve (convex)
// TODO: 0 shifting is unreversible (lossy). cant readily take the gradient. can do the gradient
//       piecewise trick like cond_rot_sigmoid
// TODO: notch rot is possibly better if not using gradient based optimization/AS. Write this as a method to connections in 
//       parallel with multiplication 
//
// NOTE: for now, multiplication is plenty fast and well optimized. Notch_rot may be more expressive.
