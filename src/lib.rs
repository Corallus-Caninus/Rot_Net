pub mod psyclones {
    //TODO: macro generic for variable kwargs and generic input types.
    //      same thing for output vector? not as important yet unless easy to implement.
    //TODO: does byte get vectorized and/or represented as a byte in modern pipelines? parameters
    //      of 64 or 32 bits will defeate the purpose of this quantization. ISAs seem to have byte
    //      operations but verify this with radare2
    use rand::*;
    use std::ops::Index;
    /// linearly approximated activation functions and their derivatives
    pub mod activations {
        //      hard code a limit to max connection in architecture search or
        //      implementation.
        //      size node edges based on address size of edges (l1 cache-line trick)

        /// returns the derivative function's output for cond_rot_act
        /// returns: g'(x) where g is cond_rot_act
        pub fn cond_rot_grad(x: u32) -> u8 {
            const SEGMENTS: [u32; 5] = [30, 75, 128, 170, 255];
            // TODO: slopes must be doubled since this creates a deriv of 1/2?
            //      this should be correct. just check back when implementing.
            //      can just use a slope of 2 and 4 why 1/2 anyways? 2 and 4 for
            //      better approximation of a given sigmoid.
            //      Theres another math problem hiding here..
            //   division of 1/2 may provide the monotonically decreasing
            //   aspect of sigmoid necessary for the inflection-concavity symmetry

            // TODO: precision slope these (epsilon slope piecewise) in a given direction
            //       e.g. into point of inflection {>,>,<,<}?
            if x < SEGMENTS[0] {
                0
            } else if x < SEGMENTS[1] {
                1
            // TODO: redundant piecewise slope not changing
            // } else if x < SEGMENTS[2] {
            //     2
            // //GAUSSIAN POINT OF CONCAVITY//
            } else if x < SEGMENTS[3] {
                2
            } else if x < SEGMENTS[4] {
                1
            } else {
                0
            }
        }

        // TODO: how important is sigmoid for normalized sum? are such features valuable in
        //       weight space (mutate piecewise without LUT)? relu and other acti funcs
        //       show that its more
        //       than a weighted normalization and is the term of approximation. start
        //       with what works and investigate empirically.
        // TODO: shift the head over into the domain.
        // TODO: analyse the sigmoid function to calculate the best fitting slope
        //       constrained by 5 segments.
        //       dont just guess. simple calculus and linear regression can resolve this.
        /// Approximate a sigmoid functions inflection and concavity features with
        /// byte shifting and mask offsetting.
        /// This implementation requires normalization of the connection parameters to
        /// prevent overflow.
        pub fn cond_rot_act(x: u8) -> u8 {
            const SEGMENTS: [u8; 5] = [30, 75, 128, 170, 255];
            //SEGMENTATION DESCRIPTION:
            // 1. tail of sigmoid is somewhat arbitrarily set @ 30 which is 10% of domain.
            // 2. head of sigmoid is clipped at 255, NOTE: sigmoid spans parameter
            //    precision not cur_buffer precision, this may change.
            // The lines and intercept are solved for two slopes: 1/2 and 2 given 1.
            // and 2. This approximation is the maximum precision and minimum error
            // optimizing for speed.
            // least this and LLVM should reorganize but error on the side of safety.
            if x < SEGMENTS[0] {
                0
            } else if x < SEGMENTS[1] {
                ((x >> 1) - 15) as u8
            } else if x < SEGMENTS[2] {
                ((x << 1) & 0x7F) as u8
            //SIGMOID POINT OF INFLECTION//
            } else if x < SEGMENTS[3] {
                ((x << 1) | 0x80) as u8
            } else if x < SEGMENTS[4] {
                ((x >> 1) | 0x80) as u8
            } else {
                255
            }
        }
    }
    ///linearly approximated connection weightings and their derivatives
    pub mod connections {
        /// weights a given signal by a connections parameter.
        ///
        /// NOTE: exponential linear approximation has alot of interesting features. 1 point of
        /// intersection 2 slopes. can approximate any given quadrant of a saddle
        /// (approximate multiplication) as well as quadratic while normalizing.
        ///
        /// BITREP:
        /// bits 8 and 1 set dropout and passthrough behaviour for network.
        /// bits 2 and 3 select function (approximation of a saddle quadrant)
        /// bits 4-7 select slope of given function.  
        ///
        /// TODO: test with radare2 if binary operations of scrubbing >><< result in
        ///      binary-compare equivalent instruction since its much faster even
        ///      with barrel shifter. Also ensure byte instructions are stuffed to mmx or used
        ///      moreover. This heavily relies on the llvm backend.
        /// TODO: may be worth unrolling as bitchecks for some level of permutations
        ///       if optimizations dont occur.
        /// TODO: skip list > comparisons like in cond_rot_act faster than shifting?
        /// MSB isnt used so consider u8/2
        /// if param > u8/4{
        /// if param > u8/8{}
        /// }
        pub fn weight(signal: u8, param: u8) -> u8 {
            const MSB_BITCHECK: u8 = 0b10000000;
            const LSB_BITCHECK: u8 = 0b00000001;
            // dont have to worry about scrubbing using bitcompare instructions.
            if param >> 7 == LSB_BITCHECK {
                //CMP BIT 8
                //passthrough the signal as a residual connection that skips this layer:
                // TODO: special case do not activate in forward_propagation! consider
                //       an option<u8> return type or just check in calling scope and
                //       pass the bit here.
                //  TODO: a node_index with 0 connections can signify a residual connection instead
                //  of here. This is usize vs 8 bit though.
                //       option<*T> == *T but does option<T> == T?
                // TODO: some computation balancing can be done by performing weighting at different
                //       depths of residual connections. instead of immediately weighting then residually passing
                //       through layers until at the out node pick a layer that isnt wide in the sequence 
                //       of residual connections and perform weighting at that position. This
                //       should be done later.
                signal
            } else if param << 7 == MSB_BITCHECK {
                //CMP BIT 1
                //deactivated connection so shunt signal:
                //TODO: allow dropout in rot_net for performance.
                0 as u8
            } else {
                //perform the weighting function:
                if param >> 1 << 7 == MSB_BITCHECK {
                    //CMP BIT 2
                    if param >> 2 << 7 == MSB_BITCHECK {
                        //CMP BIT 3
                        //Exponential
                        //pass bits 5-8 into function for slopes
                        linear_exponential(signal, param)
                    } else {
                        //Reflected Exponential
                        linear_decay_logarithm(signal, param)
                    }
                } else {
                    //reflects along y axis
                    if param >> 2 << 7 == MSB_BITCHECK {
                        //CMP BIT 3
                        //Logarithm
                        linear_logarithm(signal, param)
                    } else {
                        //Exponential Decay
                        linear_decay_exponential(signal, param)
                    }
                }
            }
        }
        //TODO: verify these dont require offsets since >><< 2 requires values > 128+64
        //      to be in phase
        //  TODO: sort these bit compares so they align for each function's concentric rectangles
        //        (read: Polygons)
        //        of intersections so mutations are done sensibly and represent a latent
        //        representation
        ///approximate an exponential function of variable time constants with two lines
        pub fn linear_exponential(param: u8, signal: u8) -> u8 {
            const MSB_BITCHECK: u8 = 0b10000000;
            // NOTE: these should all be > 127 since using upper half of domain in exponential
            const INTERCEPTS: [u8; 4] = [146, 171, 205, 219];

            // TODO: this is the only function not aligned
            //       with intercepting lines: 1,1 2,1 1,2 2,2
            //       TEST AND CLOSE
            // TODO: reorganize intercepts for readability

            if param >> 3 << 7 == MSB_BITCHECK {
                //CMP BIT 4
                // exponential with slopes 1/2,2
                //  intercept @ (170.6_,85.3_)
                if signal > INTERCEPTS[1] {
                    signal << 1
                } else {
                    signal >> 1
                }
            } else if param >> 4 << 7 == MSB_BITCHECK {
                //CMP BIT 5
                // exponential with slopes 1/2, 4
                // intercept @ (219.4285714, 109.7142857)
                if signal > INTERCEPTS[3] {
                    signal << 2
                } else {
                    signal >> 1
                }
            } else if param >> 5 << 7 == MSB_BITCHECK {
                //CMP BIT 6
                // exponential with slopes 1/4, 2
                // intercept @ (146.2857, 36.57143)
                if signal > INTERCEPTS[0] {
                    signal << 1
                } else {
                    signal >> 2
                }
            } else {
                //CMP BIT 7
                // exponential with slopes 1/4, 4
                //  intercept @ (204.8, 51.2)
                if signal > INTERCEPTS[2] {
                    signal << 2
                } else {
                    signal >> 2
                }
            }
        }
        //TODO: this isnt really a linear approximation of logarithmic decay. Call these
        //      reflections of exponent and logarithm to make more intuitive its relation to saddle
        //      function and scaling rectangular intercepts.
        ///approximate a logarithmic decay function (reflected logarithm) with 2 lines
        pub fn linear_decay_logarithm(param: u8, signal: u8) -> u8 {
            const MSB_BITCHECK: u8 = 0b10000000;
            // NOTE: these should all be > 127 since using upper half of domain in exponential
            const INTERCEPTS: [u8; 4] = [170, 219, 146, 204];

            // TODO: these are not in ascending order of decay constant. sorted with
            // every other value (this will matter when attempting to differentiate)
            if param >> 3 << 7 == MSB_BITCHECK {
                //CMP BIT 4
                // intercept @ (170, 170)
                if signal > INTERCEPTS[0] {
                    255 - (signal << 1) //NOTE: the inverse from 255
                } else {
                    255 - (signal >> 1)
                }
            } else if param >> 4 << 7 == MSB_BITCHECK {
                //CMP BIT 5
                // intercept @ (218.5714, 145.7143)
                if signal > INTERCEPTS[1] {
                    255 - (signal << 2)
                } else {
                    255 - (signal >> 1)
                }
            } else if param >> 5 << 7 == MSB_BITCHECK {
                //CMP BIT 6
                // intercept @ (145.7142, 218.5714)
                if signal > INTERCEPTS[2] {
                    255 - (signal << 1)
                } else {
                    255 - (signal >> 2)
                }
            } else {
                //CMP BIT 7
                // intercept @ (204,204)
                if signal > INTERCEPTS[3] {
                    255 - (signal << 2)
                } else {
                    255 - (signal >> 2)
                }
            }
        }
        ///approximate a logarithm with 2 lines
        /// NOTE: the slope change in this permutation
        pub fn linear_logarithm(param: u8, signal: u8) -> u8 {
            const MSB_BITCHECK: u8 = 0b10000000;
            const INTERCEPTS: [u8; 4] = [85, 109, 36, 51];

            // TODO: can offset be a bit mask? probably the same backend optimizations.
            // TODO: these are not in ascending order of decay constant. sorted with
            // every other value (this will matter when attempting to differentiate)
            if param >> 3 << 7 == MSB_BITCHECK {
                //CMP BIT 4
                // intercept @ (85, 170)
                // for offset: upper slope is y=0.5x + 127.5
                if signal > INTERCEPTS[0] {
                    (signal >> 1) + 128
                } else {
                    signal << 1
                }
            } else if param >> 4 << 7 == MSB_BITCHECK {
                //CMP BIT 5
                // intercept @ (109.2857, 218.5714)
                // for offset: upper slope is y=0.25x + 191.25
                if signal > INTERCEPTS[1] {
                    (signal >> 2) + 192
                } else {
                    signal << 1
                }
            } else if param >> 5 << 7 == MSB_BITCHECK {
                //CMP BIT 6
                // intercept @ (36.42857, 145.714285)
                // for offset: upper slope is y=0.5x + 127.5
                if signal > INTERCEPTS[2] {
                    (signal >> 1) + 128
                } else {
                    signal << 2
                }
            } else {
                //CMP BIT 7
                //intercept @ (51,204)
                // for offset: upper slope is 0.25x + 191.25
                if signal > INTERCEPTS[3] {
                    (signal >> 2) + 192
                } else {
                    signal << 2
                }
            }
        }
        ///approximate an exponential decay function with 2 lines
        pub fn linear_decay_exponential(param: u8, signal: u8) -> u8 {
            const MSB_BITCHECK: u8 = 0b10000000;
            const INTERCEPTS: [u8; 4] = [85, 109, 36, 51];

            // TODO: does this need offset considerations as well?
            //       just unittest at this point.
            if param >> 3 << 7 == MSB_BITCHECK {
                if signal > INTERCEPTS[0] {
                    //CMP BIT 4
                    // intercept @ (85, 85)
                    // for offset: under slope is y=-0.5 + 127.5
                    128 - (signal >> 1)
                } else {
                    255 - (signal << 1)
                }
            } else if param >> 4 << 7 == MSB_BITCHECK {
                //CMP BIT 5
                // intercept @ (109.2857, 36.4286)
                // for offset: under slope is y=-0.25 + 64
                if signal > INTERCEPTS[1] {
                    64 - (signal >> 2)
                } else {
                    255 - (signal << 1)
                }
            } else if param >> 5 << 7 == MSB_BITCHECK {
                //CMP BIT 6
                // intercept @ (36.42857, 109.2857)
                // for offset: under slope is y=-0.5 + 128
                if signal > INTERCEPTS[2] {
                    128 - (signal >> 1)
                } else {
                    255 - (signal << 2)
                }
            } else {
                //CMP BIT 7
                // intercept @ (51, 51)
                // for offset: under slope is y=-0.25 + 64
                if signal > INTERCEPTS[3] {
                    64 - (signal >> 2)
                } else {
                    255 - (signal << 2)
                }
            }
        }
        //TODO: derivative functions
    }

    /// an Artificial Neural Network represented as a flattened tensor of layer wise adjacency
    /// matrices. Uses shifts, conditions and masking to approximate linear functions to improve
    /// CPU based Neural Networks and provide a platform for both architecture search given finite
    /// resources and transfer learning.
    ///
    /// ---
    /// Recommended literature:
    /// --K.Stanley's NEAT algorithm for how architecture search and fitness
    /// landscape modelling can be done (genetic distance/position using innovation numbers).
    /// --fundamentals of neural networks (any author) including gradient descent and optimizers.
    /// Understand ADAM vs SGD and the limitations and performance tradeoffs in these
    /// methodologies.
    /// --batch normalization (any author). How batch normalization effects bias terms and changes
    /// the fitness landscape. The tradeoffs of batch size and the computation of the two normalization
    /// coefficients.
    /// --fundamentals of computer architecture. How pipelines work, what causes
    /// bubbles/stalls/noops. pipeline stuffing and parallel execution engines. (any author)
    pub struct rot_net {
        tensor: Vec<u8>, // a vectorized compact representation of connections
        // NOTE: this is alot of vectors but it keeps parameters tight like wonder womans lasso
        // layer widths is num nodes to chunk. Each index is a layer
        layer_widths: Vec<usize>,
        // layer nodes is the chunk size of tensor parameters for each node
        // in this layer. Each index is a node's connections (a node).
        layer_nodes: Vec<usize>,

        // counter for iteration defaults 0 NOTE: NON-ATOMIC
        layer_counter: usize, 
    }
    // NOTE: using iterators one can implement a custom counting type to address larger than
    //       architecture's usize such as 127 type but should be considered in conjunction with
    //       MMX/AVX passes in LLVM
    //
    ///  iterate this rot_net by each layers connections (parameters) as slices of parameters per
    ///  node. 
    ///  inner vectors are row span (in_connections) and entries in vec are column span (nodes)
    ///  of the current layer matrix. This returns value copies and is not for mutation.
    impl Iterator for rot_net {
        type Item = Vec<Vec<u8>>;

        //TODO: Yet Another Pointer Cleanup
        //TODO: whats the over under on cloning here..
        fn next(&mut self) -> Option<Vec<Vec<&u8>>> {
            if self.layer_counter == self.layer_widths.len() {
                self.layer_counter = 0;
                None
            } else {
                //NOTE: much of this is parallelizable

                //TODO: BEGIN REFACTOR EXTRACTION //
                // the index coordinates for this layer's nodes
                // the starting index of the layer
                let layer_index = self.layer_widths.iter().take(self.layer_counter).sum();
                // the length of the layer
                let layer_length = self
                    .layer_widths
                    .iter()
                    .skip(self.layer_counter)
                    .next()
                    .unwrap();
                // The index coordinates for each nodes connections
                // the starting node of the layer
                let node_index = self.layer_nodes.iter().take(layer_index).sum::<usize>();
                // the length of the nodes in this layer
                let node_length = self
                    .layer_nodes
                    .iter()
                    .skip(layer_index)
                    .take(*layer_length)
                    .collect::<Vec<&usize>>();
                // TODO: END REFACTOR EXTRACTION //

                let layer = node_length
                    .iter()
                    .enumerate()
                    .map(|node| {
                        self.tensor
                            .iter()
                            .skip(
                                node_index
                                    + node_length.iter().take(node.0).cloned().sum::<usize>(),
                            )
                            .take(**node.1)
                            //TODO: prove if this has better cache locality performance.
                            //.cloned()//NOTE: this should be equivalent of map copy deref. is this preferable?
                            .collect::<Vec<&u8>>()
                    })
                    .collect::<Vec<Vec<&u8>>>();
                // ready for matrix multiplication or other weighting-normalization-activation
                // routine

                self.layer_counter += 1;
                Some(layer)
            }
        }
    }
    // TODO: rename for readability and to convey teirs of abstraction
    /// Returns a node's layer index and node index in that layer.
    pub struct net_index {
        layer: usize,
        node: usize,
    }
    /// The index and length of a node's connections in self.tensor.
    pub struct node_tensor_index{
        index: usize,
        length: usize,
    }
    //TODO: derive next for layerwise iteration with itertools
    impl rot_net {
        ///return the starting index and length of a node's index in self.tensor
        fn get(&self, index: net_index) -> node_tensor_index {
            let layer_index = self.layer_widths.iter().take(index.layer).sum();

            let node_index = self.layer_nodes.iter().take(layer_index).sum::<usize>();
            let node_length = self.layer_nodes.iter().skip(layer_index).next();

            node_tensor_index{node_index, node_length}
        }
        /// rot_net constructor that builds an initial fully connected topology
        pub fn initialize_rot_net(num_inputs: usize, num_outputs: usize) -> Self {
            let mut rng = rand::thread_rng();

            //initialize the initial topology's layer.
            let mut init_topology = vec![];
            for connection in 0..num_inputs * num_outputs {
                init_topology.push(rng.gen::<u8>());
            }
            //initialize rot_net with the single layer.
            rot_net {
                tensor: init_topology,
                layer_widths: vec![num_inputs * num_outputs],
                // TODO: this should be a value for each input node
                layer_nodes: vec![num_inputs],
                layer_counter: 0,
            }
        }
        ///adds a connection by passing two addresses of positions in the vector. must be acyclic
        ///(the first layer must be a smaller index than the second).
        ///This is the fundamental mutation operation for rot_net.
        fn add_connection(&mut self, input_node: net_index, output_node: net_index) {
            //NOTE: shouldnt need output_node otherwise since nodes are positionally indexed wrt
            //      input connections.
            let intermediate_layers = input_node.layer - output_node.layer;
            debug_assert!(intermediate_layers >= 0); // cycles are not allowed this assertion should be wrapped away

            let mut rng = rand::thread_rng();

            // 1. mutate previous layer, add the in_connection to this layer.
            // parameters are represented by incoming connections to nodes so we add to output_node here.
            let new_connection =  rng.gen();
            self.tensor.insert(self.get(output_node).index, new_connection);
            
            if intermediate_layers == 0 {
                // 2. add a new connection to the node by inserting in previous layer
                //    since layers are defined as connections going into nodes.
                // NOTE: these have to be done together to preserve index in self.tensor 
                //       so consider refactoring
                let prev_value = self.layer_nodes.remove(input_node.node);
                self.layer_nodes.insert(input_node.node, prev_value+1);
            } else {
                // 3. add residual connections for all cross-sectional edges of the new layer if
                //    multiple intermediate layers exist.
                // TODO: this should always create nodes
                let tensor_node = self.get(input_node).index;
                //count from one since first layer already has actual connection.
                for layer in 1..intermediate_layers{
                    let residual_connection = 0b10000000;
                    //TODO: would self.iter() be useful here?

                    self.layer_width.iter().skip(input_node.layer+layer).next()+=1;
                    self.layer_nodes.iter().skip(input_node.layer).next()+=1;

                    self.tensor.insert(self.get(output_node).index, residual_connection);
                    //TODO: @DEPRECATED
                    //self.layer_widths[input_node.layer+layer]+=1;
                    //self.layer_nodes[input_node.node]+=1;;
                }
            }
        }
        //TODO: if a new layer is created all residual passthrough connections with the new layers
        //      cross section must be updated.
        fn add_split_node(&mut self, input_node: usize, output_node: usize) {}
        pub fn add_random_connection(&mut self) {}
        pub fn add_random_split_node(&mut self) {}
        // forward propagate through the network a vector of signals and return the output vector
        // of signals.
        //
        //Each layer in forward propagation:
        //layer_widths.pop() == 3 //this layer will have 3 nodes
        //connections = layer_nodes.take(3)
        //for connection in connections{
        //tensor.take(node)//node is the number of connections for this node in this layer.
        //
        //        pub fn forward_propagate(&self, signals: Vec<u8>) {
        //            //-> Vec<u8>
        //            let mut prev_layer = 0;
        //            // TODO: consider .chunks() to improve performance (l1 cache size)
        //            //       this proabaly happens anyways and size of each out_edge should be l1 cache
        //            self.layers.into_iter().for_each(|layer_width| {
        //                self.tensor.iter().skip(prev_layer).take(layer_width).iter();
        //                prev_layer = layer_width
        //            });
        //        }
    }
}
