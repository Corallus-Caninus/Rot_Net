//#![feature(box_into_boxed_slice)]
pub mod psyclones {
    //TODO: reduce usage of cloned and clone. ensure cache optimal at
    // least with addressing? TODO: macro generic for variable
    // kwargs and generic input types.      same thing for output
    // vector? not as important yet unless easy to implement.
    // TODO: ensure all types T can be discretized into u8.
    // dereference pointers to prevent      passing pointer as a
    // type, possibly assert. Use a macro so there isnt as much
    //      overhead for the generic operation and since topology must
    // be initialized.
    use rand::*;
    use std::fmt;
    use std::ops::Index;
    use std::string::String;
    /// linearly approximated activation functions and their
    /// derivatives
    pub mod activations {
        //      hard code a limit to max connection in architecture
        // search or      implementation.
        //      size node edges based on address size of edges (l1
        // cache-line trick)

        /// returns the derivative function's output for cond_rot_act
        /// returns: g'(x) where g is cond_rot_act
        pub fn cond_rot_grad(x: u32) -> u8 {
            const SEGMENTS: [u32; 5] = [30, 75, 128, 170, 255];
            // TODO: slopes must be doubled since this creates a deriv
            // of 1/2?      this should be correct. just
            // check back when implementing.      can just
            // use a slope of 2 and 4 why 1/2 anyways? 2 and 4 for
            //      better approximation of a given sigmoid.
            //      Theres another math problem hiding here..
            //   division of 1/2 may provide the monotonically
            // decreasing   aspect of sigmoid necessary
            // for the inflection-concavity symmetry

            // TODO: precision slope these (epsilon slope piecewise)
            // in a given direction
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

        // TODO: how important is sigmoid for normalized sum? are such
        // features valuable in       weight space (mutate
        // piecewise without LUT)? relu and other acti funcs
        //       show that its more
        //       than a weighted normalization and is the term of
        // approximation. start       with what works and
        // investigate empirically. TODO: shift the head over
        // into the domain. TODO: analyse the sigmoid function
        // to calculate the best fitting slope
        //       constrained by 5 segments.
        //       dont just guess. simple calculus and linear
        // regression can resolve this.
        /// Approximate a sigmoid functions inflection and concavity
        /// features with byte shifting and mask offsetting.
        /// This implementation requires normalization of the
        /// connection parameters to prevent overflow.
        pub fn cond_rot_act(x: u8) -> u8 {
            const SEGMENTS: [u8; 5] = [30, 75, 128, 170, 255];
            //SEGMENTATION DESCRIPTION:
            // 1. tail of sigmoid is somewhat arbitrarily set @ 30
            // which is 10% of domain. 2. head of sigmoid
            // is clipped at 255, NOTE: sigmoid spans parameter
            //    precision not cur_buffer precision, this may change.
            // The lines and intercept are solved for two slopes: 1/2
            // and 2 given 1. and 2. This approximation is
            // the maximum precision and minimum error
            // optimizing for speed.
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
    ///linearly approximated connection weightings and their
    /// derivatives
    pub mod connections {
        // what the compiler does)
        /// weights a given signal by a connections parameter.
        ///
        /// NOTE: exponential linear approximation has alot of
        /// interesting features. 1 point of intersection 2
        /// slopes. can approximate any given quadrant of a saddle
        /// (approximate multiplication) as well as quadratic while
        /// normalizing.
        ///
        /// BITREP:
        /// bits 8 and 1 set dropout and passthrough behaviour for
        /// network. bits 2 and 3 select function
        /// (approximation of a saddle quadrant)
        /// bits 4-7 select slope of given function.
        ///
        /// TODO: test with radare2 if binary operations of scrubbing
        /// >><< result in      binary-compare equivalent
        /// instruction since its much faster even
        ///      with barrel shifter. Also ensure byte instructions
        /// are stuffed to mmx or used      moreover. This
        /// heavily relies on the llvm backend. TODO: may be
        /// worth unrolling as bitchecks for some level of
        /// permutations       if optimizations dont occur.
        /// TODO: skip list > comparisons like in cond_rot_act faster
        /// than shifting? TODO: make residual comparison
        /// branchless.       (at least, most/all of this
        /// needs to be branchless dont assume MSB isnt used
        /// so consider u8/2 if param > u8/4{
        /// if param > u8/8{}
        /// }
        pub fn weight(signal: u8, param: u8) -> u8 {
            const MSB_BITCHECK: u8 = 0b10000000;
            const LSB_BITCHECK: u8 = 0b00000001;
            // dont have to worry about scrubbing using bitcompare
            // instructions.
            if param >> 7 == LSB_BITCHECK {
                //CMP BIT 8
                //passthrough the signal as a residual connection
                // that skips this layer:
                // TODO: special case do not activate in
                // forward_propagation! consider
                //       an option<u8> return type or just check in
                // calling scope and       pass the
                // bit here.  NO: Boolean is
                // represented as 8 bits in modern pipelines
                //
                // NOTE: some computation balancing can be done by
                // performing       weighting at
                // different       depths of residual
                // connections. instead of immediately
                //       weighting then residually passing
                //       through layers until at the out node pick a
                // layer that       isnt wide in the
                // sequence       of residual
                // connections and perform weighting at that
                //       position. This
                //       should be done later. work stealing with
                // rayon may make this performant
                //       anyways IFF forward prop is written correctly
                // and iterator isnt atomicly
                //       sequential.
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
        //  TODO: sort these bit compares so they align for each
        // function's  concentric rectangles (read: Polygons)
        //  of intersections so mutations are done sensibly and
        // represent  a latent representation (unused bits for
        // one expression  have recessive representation)
        ///approximate an exponential function of variable time
        /// constants with two lines
        pub fn linear_exponential(param: u8, signal: u8) -> u8 {
            const MSB_BITCHECK: u8 = 0b10000000;
            // NOTE: these should all be > 127 since using upper half
            // of       domain in exponential
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
        //TODO: this isnt really a linear approximation of
        // logarithmic decay.      Call these reflections of
        // exponent and logarithm to make more      intuitive
        // its relation to saddle function and scaling rectangular
        //      intercepts.
        ///approximate a logarithmic decay function (reflected
        /// logarithm) with 2 lines
        pub fn linear_decay_logarithm(param: u8, signal: u8) -> u8 {
            const MSB_BITCHECK: u8 = 0b10000000;
            // NOTE: these should all be > 127 since using upper half
            // of domain in exponential
            const INTERCEPTS: [u8; 4] = [170, 219, 146, 204];

            // TODO: these are not in ascending order of decay
            // constant. sorted with every other value
            // (this will matter when attempting to differentiate)
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

            // TODO: can offset be a bit mask? probably the same
            // backend optimizations. TODO: these are not
            // in ascending order of decay constant. sorted with
            // every other value (this will matter when attempting to
            // differentiate)
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

    // TODO: are these intuitive to first time readers?
    // TODO: can something be extracted to impl of these?
    //       May make more readable.
    /// A high level index of a node in the topology.
    #[derive(Clone, Copy)]
    pub struct net_index {
        pub layer: usize,
        pub node: usize,
    }
    /// The index and length of a node's connections in self.tensor.
    #[derive(Clone, Copy)]
    pub struct node_tensor_index {
        pub index: usize,
        pub length: usize,
    }
    /// an Artificial Neural Network represented as a archived tensor
    /// of layer wise irregular adjacency matrices (collumn wise
    /// ragged). Uses shifts, conditions and masking to approximate
    /// linear functions to improve superscalar performance of
    /// Artificial Neural Networks and provide a platform for both
    /// architecture search given finite resources and transfer
    /// learning. e.g.
    /// some layer n:
    /// __________
    /// |10|12| 2|
    /// |1 |2 | 3|
    /// |1 |__| 2|
    /// |__|  |__|
    ///
    /// ---
    /// Recommended literature:
    /// --K.Stanley's NEAT algorithm for how architecture search and
    /// fitness landscape modelling can be done (genetic
    /// distance/position using innovation numbers).
    /// --fundamentals of neural networks (any author) including
    /// gradient descent and optimizers. Understand ADAM vs SGD
    /// and the limitations and performance tradeoffs in these
    /// methodologies.
    /// --batch normalization (any author). How batch normalization
    /// effects bias terms and changes the fitness landscape. The
    /// tradeoffs of batch size and the computation of the two
    /// normalization coefficients.
    /// --fundamentals of computer architecture. How pipelines work,
    /// what causes bubbles/stalls/noops. pipeline stuffing and
    /// parallel execution engines. (any author)
    #[derive(Clone)] //TODO: where is this called it is potentially very costly.
    pub struct rot_net {
        tensor: Vec<u8>, /* a vectorized compact representation of
                          * connections */
        // NOTE: this is alot of vectors but it keeps parameters
        // (which scale the most) tight like wonder womans lasso
        // layer widths is num nodes to chunk. Each index is number
        // of nodes in that layer.
        layer_widths: Vec<usize>, //scales with number of layers
        // layer nodes is the chunk size of tensor parameters for
        // each node in this layer. Each entry is the number
        // of connections in a node.
        layer_nodes: Vec<usize>, //scales with number of nodes

        // NON-ATOMIC ITERATION LOCALS //
        // counter for iteration defaults 0 NOTE: NON-ATOMIC
        layer_counter: usize,
        /*TODO: index in self.tensor to reduce round trip summing
         * layer_index: usize, */
    }
    // TRAITS //
    // NOTE: using iterators one can implement a custom counting type
    // to address larger than       architecture's usize such as
    // 127 bit type but should be considered in conjunction with
    //       MMX/AVX passes in LLVM. This kind of free runtime feature
    // is awesome! I want to be a       part of this.
    //
    // TODO: DoubleEndedIterator for backprop
    ///  iterate this rot_net by each layers connections (parameters)
    /// as slices of parameters per  node. inner vectors are row
    /// span (in_connections) and entries in vec are column span
    /// (nodes)  of the current layer matrix. This returns value
    /// copies and is not for mutation.
    impl Iterator for rot_net {
        type Item = Vec<Box<[u8]>>;

        fn next(&mut self) -> Option<Vec<Box<[u8]>>> {
            if self.layer_counter == self.layer_widths.len() {
                self.layer_counter = 0;
                None
            } else {
                //NOTE: much of this is parallelizable
                //TODO: can intermediate self local types help with
                // sum overhead?      static usize in
                // self almost dont matter at all at scale.
                //      store position in self.tensor at each
                // iteration? layer_count, layer_index?
                //      just add current layer length to layer index
                // instead of round trip sum each
                //      time.
                //  is it possible for the compiler to balance memory
                // and cpu resources with  automatic
                // memorization using the polyhedral loop model and
                // "source insertion"  technique?

                // the index coordinates for this layer's nodes
                // the starting index of the layer
                let layer_index = self
                    .layer_widths
                    .iter()
                    .take(self.layer_counter)
                    .sum();
                // the length of the layer
                let layer_length = self
                    .layer_widths
                    .iter()
                    .skip(self.layer_counter)
                    .next()
                    .unwrap();
                // The index coordinates for each nodes connections
                // the starting node of the layer
                let node_index = self
                    .layer_nodes
                    .iter()
                    .take(layer_index)
                    .sum::<usize>();
                // the length of the nodes in this layer
                let node_length = self
                    .layer_nodes
                    .iter()
                    .skip(layer_index)
                    .take(*layer_length)
                    .collect::<Vec<&usize>>();

                let mut layer = node_length
                    .iter()
                    .enumerate()
                    .map(|node| {
                        self.tensor
                            .iter()
                            .skip(
                                node_index
                                    + node_length
                                        .iter()
                                        .take(node.0)
                                        .cloned()
                                        .sum::<usize>(),
                            )
                            .take(**node.1)
                            .cloned()
                            //TODO: prove if this has better cache
                            //locality performance.
                            .collect::<Vec<u8>>()
                            .as_slice()
                            .into()
                    })
                    .collect::<Vec<Box<[u8]>>>();
                //TODO: these should all be slices
                //      because these should never be modified.
                //      trait doesnt allow such reference as
                //      slice so implement Box slice.
                //      Box doesnt implement copy so cant into.
                //      may be worth a RefCell but should investigate
                //      nightly experimental features further.
                //      also can [[u8]] be ragged not unless outer is
                // [&]

                self.layer_counter = self.layer_counter + 1;
                //since vector is immutable and is a ragged slice
                layer.shrink_to_fit();
                let layer = layer;

                // ready for matrix multiplication or other
                // weighting-normalization-activation
                // routine
                Some(layer)
            }
        }
    }
    // used for debuging
    // TODO: impl Display for rot_net{} for debugging
    /// Print out the layer wise structure of the Rot_Net
    impl fmt::Display for rot_net {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            let mut network = self.clone();
            let mut printout = "\n__layer__\n".to_owned();
            printout.push_str("Node Id: [Connections,] \n\n");

            network.enumerate().for_each(|(index, layer)| {
                printout.push_str("\n___");
                printout
                    .push_str(&String::from(format!("{}", index)));
                printout.push_str("___\n");
                for (index, node) in layer.iter().enumerate() {
                    if index > 0 && index < layer.iter().len() {
                        printout.push('\n');
                    }
                    printout.push_str(&String::from(format!(
                        "{}: ",
                        index
                    )));
                    for (index, connection) in node.iter().enumerate()
                    {
                        printout.push_str(&String::from(format!(
                            "{},",
                            connection
                        )));
                    }
                }
            });
            printout.push_str("\n___\n");
            write!(f, "{}", printout)
        }
    }

    // METHODS //
    impl rot_net {
        ///return the starting index and length of a node's
        /// parameters (input connections) in self.tensor
        /// given an abstract network index
        fn get(&self, index: net_index) -> node_tensor_index {
            //TODO: BUG layer error here and iterator.
            // where this node is in self.layer_nodes
            let node_index = self
                .layer_widths
                .iter()
                .take(index.layer)
                .sum::<usize>();
            // where this connection is in self.tensor
            let index = self
                .layer_nodes
                .iter()
                .take(node_index)
                .sum::<usize>();

            //TODO: TEST
            println!("translating a network index into a tensor archive index..");
            println!(
                "with self.layer_widths: {}",
                self.layer_widths.len()
            );
            println!(
                "with self.layer_nodes: {}",
                self.layer_nodes.len()
            ); // these are correct.
            println!("with connections: {}", self.tensor.len());
            // these are correct.

            println!("with node_index: {}", node_index);
            println!("with index: {}", index);
            //TODO: END OF TEST

            // the number of connections in this node
            let connection_length =
                *self.layer_nodes.iter().skip(index).next().unwrap();

            node_tensor_index {
                index: index,
                length: connection_length,
            }
        }
        /// rot_net constructor that builds an initial fully connected
        /// topology
        pub fn initialize_rot_net(
            num_inputs: usize,
            num_outputs: usize,
        ) -> Self {
            //TODO: rework data structure to have a null layer as
            // output nodes.
            //
            let mut rng = rand::thread_rng();

            // initialize the initial topology's layer.
            let mut init_topology = vec![];
            for connection in 0..num_inputs * num_outputs {
                init_topology.push(rng.gen::<u8>());
            }
            //initialize rot_net with the single fully connected
            // layer.
            rot_net {
                tensor: init_topology,
                layer_widths: vec![num_inputs + num_outputs], /* just 1 initial layer */
                // a vector of length num_inputs with each value set
                // to num_outputs
                layer_nodes: vec![num_outputs; num_inputs],
                layer_counter: 0,
            }
        }
        ///adds a connection by passing two network indices.
        ///must be acyclic.
        ///connection weights are initialized as random.
        pub fn add_connection(
            &mut self,
            input_node: net_index,
            output_node: net_index,
        ) {
            // TODO: need to check MSB of connection when performing
            // add_node to       ensure it is not a
            // residual connection. Can only assert here
            // NOTE: shouldnt need output_node otherwise since nodes
            // are      positionally indexed wrt input
            // connections.
            println!("adding connection with output layer: {} and input layer: {}", output_node.layer, 
                input_node.layer);
            let intermediate_layers =
                output_node.layer - input_node.layer;
            println!("intermediate layers: {}", intermediate_layers);
            // cycles are not allowed this assertion should be wrapped
            // away
            debug_assert!(intermediate_layers >= 0);

            // create a non-residual connection parameter
            let mut rng = rand::thread_rng();
            let new_connection = rng.gen::<u8>() & 0b01111111;

            // add residual connections for all cross-sectional edges
            // of the new layer if multiple intermediate
            // layers exist. iterate up to one less since
            // last layer has actual connection
            // add a phantom node for each layer that cross sects
            // (traces) the residual connection up to the
            // actual connection. a phantom node always
            // exists for all layers except the fully connected
            // initial layer. find the phantom node.
            const residual_connection: u8 = 0b10000000;
            for layer in input_node.layer..output_node.layer - 1 {
                //update the layer_nodes for the new residual
                // connection
                let layer_start = self
                    .layer_widths
                    .iter()
                    .take(layer)
                    .sum::<usize>();
                let layer_length = self
                    .layer_widths
                    .iter()
                    .skip(layer_start)
                    .next()
                    .unwrap();

                let node_start = self
                    .layer_nodes
                    .iter()
                    .take(layer_start)
                    .sum::<usize>();
                let node_length = self
                    .layer_nodes
                    .iter()
                    .skip(layer_start)
                    .next()
                    .unwrap()
                    .clone();

                let layer_connections = self
                    .tensor
                    .iter()
                    .skip(node_start)
                    .take(node_length)
                    .collect::<Vec<&u8>>();

                let mut phantom_node_index = layer_connections
                    .iter()
                    .enumerate()
                    .filter(|x| **x.1 == 0b10000000)
                    .map(|x| x.0)
                    .collect::<Vec<usize>>();
                phantom_node_index.sort();
                // the location of the first connection index of the
                // phantom node
                let phantom_node_index = phantom_node_index[0];

                //update layer_node connection count for the
                // phantom_node
                let mut phantom_index = layer_start;
                let mut cur_connection_index = 0;
                for node in self.layer_nodes.iter().skip(layer_start)
                {
                    if cur_connection_index == phantom_node_index {
                        // this node is this layers phantom_node
                        break;
                    }
                    // sum the connection count for each node until we
                    // reach the phantom node's connection index
                    cur_connection_index += node;
                    phantom_index += 1;
                }
                let mut prev_phantom_value =
                    self.layer_nodes.remove(phantom_index);
                prev_phantom_value += 1;
                self.layer_nodes
                    .insert(phantom_index, prev_phantom_value);

                //insert residual_connection in phantom_node
                self.tensor.insert(
                    self.get(output_node).index + intermediate_layers,
                    residual_connection,
                );
            }
            // add the new connection at the end of the residual
            // connection trace TODO: need to update node
            // connection value for this node.
            // TODO: extract these into method during post-test and
            // debug refactor. TODO: automatic refactoring
            // would be awesome implementation of GPT3
            let node_index = self
                .layer_widths
                .iter()
                .take(output_node.layer)
                .sum::<usize>();

            let prev_node_connections =
                self.layer_nodes.remove(node_index);
            let node_connections = prev_node_connections + 1;
            self.layer_nodes.insert(node_index, node_connections);
            self.tensor.insert(
                self.get(output_node).index + intermediate_layers,
                new_connection,
            );
        }
        //TODO: if a new layer is created all residual passthrough
        // connections      with the new layers cross section
        // must be updated.
        /// adds two new connections, one of which is a new node that
        /// may be in a new layer. The connections are
        /// initialized as random. NOTE: if deactivating the
        /// split connection as per K.Stanley this
        ///       breaks representation (should only randomize
        /// incoming connection out connection       is 1 in
        /// this case).

        pub fn add_split_node(
            &mut self,
            input_node: net_index,
            output_node: net_index,
        ) {
            //TODO: indicate which side of a layer a node is on? may
            // need data structure rework      can use
            // iterator and ragged tensor to ensure connections arent
            // cyclic.      how can connections be used to
            // ensure always acyclic? pass in node_tensor_index
            //      with a sub index for connection and split
            // accordingly?
            let mut rng = rand::thread_rng();
            println!("--NODE CREATION ROUTINE:");

            let intermediate_layers =
                input_node.layer - output_node.layer;

            // generate a random non-residual connection
            let in_connection = rng.gen::<u8>() & 0b01111111;
            let out_connection = rng.gen::<u8>() & 0b01111111;
            // create a new layer
            // increment index for self.layer_widths to fill later
            // insertion gap.
            let new_node = net_index {
                layer: input_node.layer + 1,
                node: 1,
            };

            //determine if this node defines a new layer.
            if intermediate_layers == 0 {
                println!("creating a new layer for this node..");
                // this node is a new layer
                //TODO: this layer requires the same length as the
                // previous layer self.layer_widths.
                // insert(input_node.layer + 1, 1);
                // TODO: node's out_connection isnt placed in new
                // layer self.layer_nodes.
                // insert(input_node.node + 1, 0);

                //LAYER CREATION ROUTINE
                // TODO: the new layer is inserted behind the previous
                // layer

                // TODO: there is an extractable method here..
                // layer index
                let layer = self
                    .layer_widths
                    .iter()
                    .take(input_node.layer)
                    .sum::<usize>();
                // tensor index
                let layer_start = self
                    .layer_nodes
                    .iter()
                    .take(layer)
                    .sum::<usize>();
                let layer_nodes = self
                    .layer_nodes
                    .iter()
                    .skip(layer)
                    .next()
                    .unwrap()
                    .clone();
                // tensor slice
                let tensor_slice = self
                    .tensor
                    .iter()
                    .skip(layer_start)
                    .take(layer_nodes);
                let tensor_length = tensor_slice.len();
                // now call add_connection but with residual
                // connections for all tensor_length
                let residual_tensor_length = [0..tensor_length];
                residual_tensor_length.iter().enumerate().for_each(
                    |(index, connection)| {
                        let param = 0b10000000 as u8;
                        self.tensor
                            .insert(layer_start + index, param);
                    },
                );
                self.layer_widths.insert(layer_start, 1);
                //new layer is now created
                // residual nodes are normal nodes but created with
                // only residual connections
                // and filtered in add_node and add_connection
                // mutations so they remain residual
                // until depleted from add_split_node operations on
                // their layer.
                self.layer_nodes.insert(layer, tensor_length);

                //add residual connections

                // add number of phantom nodes for residual
                // connections residual connections
                // are removed in layer existing condition and created
                // in new layer condition since node
                // additions are the only way to change a residual
                // connection.
                //  TODO: Can represent all residual connections with
                // residual nodes     (connection
                // length 0) and search for residual nodes when
                // creating a new node. If
                //     intermediate_layers > 0 && some node has length
                // 0 remove a residual     connection
                // at that node and create a new node in that layer
                // else perform     this routine.
                //
                // END OF LAYER CREATION ROUTINE

                //add input connection to split node
                self.add_connection(input_node, new_node);
                //add output connection to split node
                self.add_connection(new_node, output_node);
            } else {
                println!("processing existing layer..");
                // a layer already exists for this node.
                // insert new node in the layer between these two
                // nodes increment the layers node
                // count
                let mut insertion_layer =
                    self.layer_widths.remove(output_node.layer);
                //increase node count for this layer
                insertion_layer += 1;
                self.layer_widths
                    .insert(output_node.layer, insertion_layer);
                let cur_node_index = self
                    .layer_widths
                    .iter()
                    .take(output_node.layer)
                    .sum::<usize>();
                self.layer_nodes.insert(cur_node_index, 0); //connections will initialize the node

                self.add_connection(input_node, new_node);
                self.add_connection(output_node, new_node);
            }
            println!("--NODE CREATION COMPLETE.");
        }
        //pub fn add_random_connection(&mut self) {
        //TODO: just call add_connection with two random samples
        //      where first is > second
        //}
        //pub fn add_random_split_node(&mut self) {
        // TODO: same as random_connection but connection must exist
        // TODO: need a way to check connections in ragged matrix and
        //       through residual connections in tensor of ragged
        // matrix
        //}
        //pub fn forward_propagate(){}
        // forward propagate through the network a vector of signals
        // and return the output vector of signals.
        //
        //Each layer in forward propagation:
        //layer_widths.pop() == 3 //this layer will have 3 nodes
        //connections = layer_nodes.take(3)
        //for connection in connections{
        //tensor.take(node)//node is the number of connections for
        //                 //this node in this layer.
        //
        //        pub fn forward_propagate(&self, signals: Vec<u8>) {
        //            //-> Vec<u8>
        //            let mut prev_layer = 0;
        //            // TODO: consider .chunks() to improve
        // performance            (l1 cache size)
        //            //       this proabaly happens anyways and size
        // of each            out_edge should be l1 cache
        //            self.layers.into_iter().for_each(|layer_width| {
        //
        // self.tensor.iter().skip(prev_layer).take(layer_width).
        // iter();                prev_layer = layer_width
        //            });
        //        }
        //
        //
        //pub fn stochastic_back_propagate(){}
        //backwards propagate through the network changing weights
        // accordingly. TODO: extract this rant to DEV_DOC.md
        //NOTE: this is not SGD. The gradient sets a scalar to a
        // distributions      variance which in turn allows
        // changes to the weight.      This ensures
        // exploration during MCTS (PoM/RoM) while still being
        //      more reasonable than the brute force+combinatorics
        // psuedo ("natural")      gradient of crossover (also
        // not a swarm optimizer so sample      and/or memory
        // (SAR map) efficient). This also allows multiple
        //      point mutations throughout the network at once,
        //      preserving solutions and increasing
        //      terms of approximation where necessary until an
        // information      equillibrium is broken in the
        // topology and the previously      optimal local
        // subtrees in the graph are then underperforming,
        //      allowing informative complexification
        // (generalization/abstraction). NOTE: This doesnt
        // solve the multi optimization problem of architecture search
        //      in itself. This is more of a sample efficiency
        // solution. multi-optimization      can be solved by
        // directly changing the topology using the gradient
        //      information for
        //      connection weight and taking a higher order derivative
        // with the      gradient for complexification (but
        // still using a scalar      variation to prevent
        // jitter      in sample space causing irreversable
        // complexification      runaway (the bane of NEAT
        // which can lead to compounding errors      in the
        // f(fitness_landscape, complexification) pathing)).
        // NOTE: the above statements lead to the corallaries that:
        //      1. The weights and complexifications must be updated
        //      stochastically but using the gradient. Using the
        // gradient      as a scalar to variation of a
        // mutation distribution may lead to      efficient
        // sampling with robust exploration.      2. The
        // mutation of weights can be performed randomly but
        //         this wastes the information processed in the
        // gradient calculation.         It is hypothesized
        // that the order of mutation/complexification in
        //         the topology is sufficient for robust exploration
        // to the extent         of global minima. Allowing
        // the distribution of         point mutation (not to
        // be confused with PoM data structure)
        //         selection with the gradient to be sufficient for
        // exploration         equivalent to genetic/swarm
        // gradient. 1. calculate back_prop error gradient
        //LOOP UNTIL INPUT NODES:
        //  2. iterate back one layer and for each connection:
        //     3. sample and conditionally perform mutation_weight
        // distribution based on error*variance     4. sample
        // and conditionally perform mutation_connection distribution
        // based on error*(variance d/dx)        node
        // selection also calculated from error*(variance d/dx) of
        // other        connections possibly connecting to
        // high performing nodes (average
        //        input_connection error). what does hebbian
        // plasticity say about this? can use        locality
        // with average error (within same subtree) for fire-together
        //        wire-together (distance-fitness selection metric).
        // If a connection is        underperforming look to
        // integrate terms.     5. sample and conditionally
        // perform mutation_node distribution based on
        // error*(1/variance d/dx)        directly on the
        // connection in question. if a connection is performing well
        // add        terms.
        //  6. return
        //  such that mutation_weight_rate > mutation_connection_rate
        // > mutation_node_rate loop can be implemented as BFS
        // that halts a subtree backprop when the gradient is not
        // taken. prefered to traverse the entire tree for
        // simplicity and thoroughness. Sometimes there is
        // a chaotic nature to the system that can be changed from a
        // shallower point in the network (why does the
        // backprop gradient weigh more on the deeper connection
        // weights. This cant possibly be condusive to
        // generalization and hints at a chaotic system seeded from
        // inputs) The sparsity of the chaotic search is what
        // makes ANN robust to high dimension data and
        // functions allowing for generalization (transfer learning)
        // and architecture reuse across domains.

        //TODO: trace split_depths tree for innovation assignment by
        // iterating rot_net     and getting
        //     shortest to longest residual connection spans. dont
        // store in data     structure since at
        //     least usize innovation per parameter. This is only
        // useful if not     doing backprop search.
        //
        //  TODO: initialize with limitation parameters as
        // configuration consts.        force usize to be
        // maximum connection per node and maximum network size in
        // search.  TODO: PoM will also be limited by this but
        // use agglomerative clustering to shrink the
        //  RoM when some threshold of PoMs is reached. This allows
        // continuous exploring of the RoM until
        //  some equillibrium RoM representation is reached for a
        // given search distance. varying  the search distance
        // over time such as epsilon decay or proportional epsilon for
        // PoM  count may be interesting. The goal is that
        // this will be as robust as SARSA but with
        //  some compression through learning. Knowledge based
        // compression as well as mapping  the model and not
        // the specific domain allows scale out to transfer learning
        // across  domains and ranges (e.g.: sensors and
        // fitness functions).
    }
}
