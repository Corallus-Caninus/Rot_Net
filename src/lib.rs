//#![feature(box_into_boxed_slice)]
//TODO: reduce usage of cloned and clone. ensure cache
// optimal at least with addressing? TODO: macro generic for
// variable kwargs and generic input types.      same thing
// for output vector? not as important yet unless easy to
// implement. TODO: ensure all types T can be discretized
// into u8. dereference pointers to prevent      passing
// pointer as a type, possibly assert. Use a macro so there
// isnt as much      overhead for the generic operation and
// since topology must be initialized.
pub mod psyclones {
    use itertools::Itertools;
    use rand::prelude::*;
    use rand::*;
    use rayon::prelude::*;
    use std::cell::RefCell;
    use std::cmp::Ordering;
    use std::fmt;
    use std::ops::Index;
    use std::string::String;
    use std::sync::{Arc, Mutex, RwLock, Weak};

    // NETWORK FUNCTIONS //
    /// linearly approximated activation functions and their
    /// derivatives
    pub mod activations {
        //      hard code a limit to max connection in
        // architecture search or
        // implementation.      size node edges
        // based on address size of edges (l1
        // cache-line trick)

        /// returns the derivative function's output for
        /// cond_rot_act returns: g'(x) where g is
        /// cond_rot_act
        pub fn cond_rot_grad(x: u32) -> u8 {
            const SEGMENTS: [u32; 5] = [30, 75, 128, 170, 255];
            // TODO: slopes must be doubled since this
            // creates a deriv of 1/2?      this
            // should be correct. just
            // check back when implementing.      can just
            // use a slope of 2 and 4 why 1/2 anyways? 2 and
            // 4 for      better approximation
            // of a given sigmoid.      Theres
            // another math problem hiding here..
            //   division of 1/2 may provide the
            // monotonically decreasing   aspect
            // of sigmoid necessary
            // for the inflection-concavity symmetry

            // TODO: precision slope these (epsilon slope
            // piecewise) in a given direction
            //       e.g. into point of inflection
            // {>,>,<,<}?
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

        // TODO: how important is sigmoid for normalized
        // sum? are such features valuable in
        // weight space (mutate piecewise without
        // LUT)? relu and other acti funcs
        //       show that its more
        //       than a weighted normalization and is the
        // term of approximation. start       with
        // what works and investigate empirically.
        // TODO: shift the head over
        // into the domain. TODO: analyse the sigmoid
        // function to calculate the best fitting
        // slope       constrained by 5 segments.
        //       dont just guess. simple calculus and linear
        // regression can resolve this.
        /// Approximate a sigmoid functions inflection and
        /// concavity features with byte shifting
        /// and mask offsetting. This implementation
        /// requires normalization of the connection
        /// parameters to prevent overflow.
        pub fn cond_rot_act(x: u8) -> u8 {
            const SEGMENTS: [u8; 5] = [30, 75, 128, 170, 255];
            //SEGMENTATION DESCRIPTION:
            // 1. tail of sigmoid is somewhat arbitrarily
            // set @ 30 which is 10% of domain.
            // 2. head of sigmoid is clipped at
            // 255, NOTE: sigmoid spans parameter
            //    precision not cur_buffer precision, this
            // may change. The lines and
            // intercept are solved for two slopes: 1/2
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
    ///linearly approximated connection weightings and
    /// their derivatives
    pub mod weights {
        // what the compiler does)
        /// weights a given signal by a weights parameter.
        ///
        /// NOTE: exponential linear approximation has alot
        /// of interesting features. 1 point of
        /// intersection 2 slopes. can approximate
        /// any given quadrant of a saddle
        /// (approximate multiplication) as well as
        /// quadratic while normalizing.
        ///
        /// BITREP:
        /// bits 8 and 1 set dropout and passthrough
        /// behaviour for network. bits 2 and 3
        /// select function (approximation of a
        /// saddle quadrant) bits 4-7 select slope
        /// of given function.
        ///
        /// TODO: test with radare2 if binary operations of
        /// scrubbing >><< result in
        /// binary-compare equivalent instruction
        /// since its much faster even
        ///      with barrel shifter. Also ensure byte
        /// instructions are stuffed to mmx or used
        /// moreover. This heavily relies on the
        /// llvm backend. TODO: may be
        /// worth unrolling as bitchecks for some level of
        /// permutations       if optimizations dont occur.
        /// TODO: skip list > comparisons like in
        /// cond_rot_act faster than shifting? TODO:
        /// make residual comparison branchless.
        /// (at least, most/all of this needs to be
        /// branchless dont assume MSB isnt used
        /// so consider u8/2 if param > u8/4{
        /// if param > u8/8{}
        /// }
        pub fn weight(signal: u8, param: u8) -> u8 {
            const MSB_BITCHECK: u8 = 0b10000000;
            const LSB_BITCHECK: u8 = 0b00000001;
            // dont have to worry about scrubbing using
            // bitcompare instructions.
            if param >> 7 == LSB_BITCHECK {
                //CMP BIT 8
                // TODO: this is residual connection
                //
                // NOTE: some computation balancing can be
                // done by performing
                // weighting at different
                // depths of residual
                // weights. instead of immediately
                //       weighting then residually passing
                //       through layers until at the out
                // node pick a layer that
                // isnt wide in the sequence
                // of residual weights and
                // perform weighting at that
                //       position. This
                //       should be done later. work stealing
                // with rayon may make this
                // performant       anyways
                // IFF forward prop is written correctly
                // and iterator isnt atomicly
                //       sequential.
                signal
            } else if param << 7 == MSB_BITCHECK {
                //CMP BIT 1
                //deactivated connection so shunt signal:
                //TODO: allow dropout in rot_net for
                // performance.
                0 as u8
            } else {
                //perform the weighting function:
                if param >> 1 << 7 == MSB_BITCHECK {
                    //CMP BIT 2
                    if param >> 2 << 7 == MSB_BITCHECK {
                        //CMP BIT 3
                        //Exponential
                        //pass bits 5-8 into function for
                        // slopes
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
        //  TODO: sort these bit compares so they align for
        // each function's  concentric rectangles
        // (read: Polygons)  of intersections so
        // mutations are done sensibly and represent
        // a latent representation (unused bits for
        // one expression  have recessive representation)
        ///approximate an exponential function of variable
        /// time constants with two lines
        pub fn linear_exponential(param: u8, signal: u8) -> u8 {
            const MSB_BITCHECK: u8 = 0b10000000;
            // NOTE: these should all be > 127 since using
            // upper half of       domain in
            // exponential
            const INTERCEPTS: [u8; 4] = [146, 171, 205, 219];

            // TODO: this is the only function not aligned
            //       with intercepting lines: 1,1 2,1 1,2
            // 2,2       TEST AND CLOSE
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
        // exponent and logarithm to make more
        // intuitive its relation to saddle function
        // and scaling rectangular      intercepts.
        ///approximate a logarithmic decay function
        /// (reflected logarithm) with 2 lines
        pub fn linear_decay_logarithm(param: u8, signal: u8) -> u8 {
            const MSB_BITCHECK: u8 = 0b10000000;
            // NOTE: these should all be > 127 since using
            // upper half of domain in
            // exponential
            const INTERCEPTS: [u8; 4] = [170, 219, 146, 204];

            // TODO: these are not in ascending order of
            // decay constant. sorted with every
            // other value (this will matter
            // when attempting to differentiate)
            if param >> 3 << 7 == MSB_BITCHECK {
                //CMP BIT 4
                // intercept @ (170, 170)
                if signal > INTERCEPTS[0] {
                    255 - (signal << 1) //NOTE: the inverse
                                        // from 255
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

            // TODO: can offset be a bit mask? probably the
            // same backend optimizations. TODO:
            // these are not in ascending order
            // of decay constant. sorted with
            // every other value (this will matter when
            // attempting to differentiate)
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
                // for offset: upper slope is y=0.25x +
                // 191.25
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
        ///approximate an exponential decay function with 2
        /// lines
        pub fn linear_decay_exponential(param: u8, signal: u8) -> u8 {
            const MSB_BITCHECK: u8 = 0b10000000;
            const INTERCEPTS: [u8; 4] = [85, 109, 36, 51];

            // TODO: does this need offset considerations as
            // well?       just unittest at this
            // point.
            if param >> 3 << 7 == MSB_BITCHECK {
                if signal > INTERCEPTS[0] {
                    //CMP BIT 4
                    // intercept @ (85, 85)
                    // for offset: under slope is y=-0.5 +
                    // 127.5
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

    // NETWORK PARAMETERS //
    /// an edge in the network graph
    #[derive(Clone, Debug)]
    pub struct connection {
        // the ID of the node this connection goes to.
        // TODO: Arc RwLock is slow and bloated.
        //       would prefer data parallel locality instead
        //       if this is only addressing based solution

        // TODO: get rid of this Arc and use data-parallelism
        // TODO: this should be weak but lives as long as self
        //       since nodes arent pruned
        output_node: Weak<RwLock<node>>,
        // the ID of this connection
        // this is currently used in place of input_node pointers
        // because it can convey topology mapping and serves the
        // same operation in forward_propagation
        // TODO: we are split depth processing for these to alleviate bloat
        // innovation: usize,
        param: u8,
    }

    /// a vertex in the network graph
    #[derive(Clone, Debug)]
    pub struct node {
        // the ID of this node
        // NOTE: would prefer for this to be an address of a connection vector but is too dynamic
        id: usize,
        // the connections going out from this node
        connections: Vec<Arc<connection>>,
        // NOTE: with these we are optimal fast for pointer-graph traversal.
        //the connections going into this node.
        //the reference lifetime is tied to the value since an edge
        // always exists in both vertices.
        input_connections: Vec<Weak<connection>>,
    }
    // impl Ord for node {
    //     fn cmp(&self, other: &Self) -> Ordering {
    //         self.id.cmp(&other.id)
    //     }
    // }
    // impl PartialOrd for node {
    //     fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
    //         Some(self.cmp(other))
    //     }
    // }
    // impl PartialEq for node {
    //     fn eq(&self, other: &Self) -> bool {
    //         self.id == other.id
    //     }
    // }
    // impl Eq for node {} //Implied by PartialEq

    /// an Artificial Neural Network represented as a sparse
    /// index format. This data structure supports both DAG
    /// (directed acyclic graphs) and DCGs (directed
    /// cyclic graphs) due to the halting algorithm.
    /// parallel edges are not supported but loops and
    /// cycles are (self connecting nodes and
    /// recurrent connections). By definition residual
    /// "skip" connections are also allowed
    ///
    ///NOTE: initial topology is first inputs+outputs
    /// nodes. sorted by most recent mutations which can
    /// be helpful for crossover or split_depth tree
    /// tracing. would likely sort these to be in the
    /// order of forward propagation to minimize cycles
    /// of the network while delaying node activation.
    ///
    /// ---Recommended literature:
    /// --K.Stanley's NEAT algorithm for how architecture
    /// search and fitness landscape modelling can be
    /// done (genetic distance/position using innovation
    /// numbers). --fundamentals of neural networks (any
    /// author) including gradient descent and
    /// optimizers. Understand ADAM vs SGD
    /// and the limitations and performance tradeoffs in
    /// these methodologies.
    /// --batch normalization (any author). How batch
    /// normalization effects bias terms and changes the
    /// fitness landscape. The tradeoffs of batch size
    /// and the computation of the two normalization
    /// coefficients. --fundamentals of computer
    /// architecture. How pipelines work, what causes
    /// bubbles/stalls/noops. pipeline stuffing and
    /// parallel execution engines. (any author)
    #[derive(Clone)] //TODO: where is this called it is potentially very
                     // costly.
    pub struct rot_net {
        pub tensor: Vec<Arc<RwLock<node>>>,
        // these are convenience variables
        // (they take relatively no space at scale)
        // the output node id's
        pub outputs: Vec<usize>,
        // the input node id's
        pub inputs: Vec<usize>,
    }
    // METHODS //
    impl rot_net {
        pub fn initialize_network(
            inputs: usize,
            outputs: usize,
        ) -> Self {
            let mut rng = rand::thread_rng();
            let mut tensor = vec![];

            let mut output_vector = vec![];
            let mut input_vector = vec![];

            // add the output connections one time
            // TODO: rework this order of operations for
            // readability
            for output in 0..outputs {
                let mut new_node = node {
                    id: output,
                    connections: vec![],
                    input_connections: vec![],
                };
                // TODO: ensure this aligns correctly with input node creation
                // one input connection innovation each
                // for input in 0..inputs {
                //     new_node
                //         .input_connections
                //         .push(innovation_counter);
                // }
                tensor.push(Arc::new(RwLock::new(new_node)));
                output_vector.push(output);
            }

            // since we are considering recurrent
            // connections as buffer roll over
            // (not param associated so doesnt bloat)
            // we allow output nodes to exist in the
            // tensor. This is +1 timestep recurrence.
            for input in 0..inputs {
                println!("initializing with id {}", outputs + input);
                // TODO: input nodes are wrong
                let mut new_node = Arc::new(RwLock::new(node {
                    id: input + outputs,
                    connections: vec![],
                    input_connections: vec![],
                }));
                input_vector.push(input + outputs);
                for output in 0..outputs {
                    let new_connection = Arc::new(connection {
                        output_node: Arc::downgrade(&tensor[output]),
                        param: rng.gen::<u8>(),
                    });
                    println!(
                        "initializing connection for node with id {}",
                        tensor.len()
                    );
                    //TODO: now push to input_connections
                    tensor[input]
                        .write()
                        .unwrap()
                        .input_connections
                        .push(Arc::downgrade(&new_connection));
                    new_node
                        .write()
                        .unwrap()
                        .connections
                        .push(new_connection);
                }
                tensor.push(new_node);
            }
            rot_net {
                tensor: tensor,
                inputs: input_vector,
                outputs: output_vector,
            }
        }
        // TODO: dont use lookup by ID (costly since data structure rework)

        /// returns a mutable reference to a node in the
        /// network given the node id
        // pub fn get_node_mut(&mut self, index: usize) -> &mut node<'c> {
        //     // TODO: output nodes arent considered in this
        //     self.tensor
        //         .par_iter_mut()
        //         .find_any(|node| node.borrow().id == index)
        //         .unwrap().borrow_mut()
        // }
        /// returns an immutable reference to a node in the
        /// network given the node id
        pub fn get_node(&self, index: usize) -> Arc<RwLock<node>> {
            // TODO: output nodes arent considered in this
            self.tensor
                .iter()
                .find(|node| node.read().unwrap().id == index)
                .unwrap()
                .clone()
        }
        // @DEPRECATED
        // TODO: this be slow
        // input_nodes in connections prove this with unittests.
        // input_nodes would also help with recurrent connections.
        // pub fn get_in_connections(
        //     &self,
        //     node_id: usize,
        // ) -> Vec<&connection> {
        //     self.tensor
        //         .par_iter()
        //         .flat_map(|node| {
        //             node.connections
        //                 .iter()
        //                 //.map(|connection| connection.output_node)
        //                 .filter(|connection| {
        //                     connection.output_node.id == node_id
        //                 })
        //                 .collect::<Vec<&connection>>()
        //         })
        //         .collect()
        // }
        /// add a connection to the network with randomized
        /// parameter
        pub fn add_connection(
            &mut self,
            input_node_id: usize,
            output_node_id: usize,
        ) {
            // println!(
            //     "adding connection..{} {}",
            //     input_node_id, output_node_id
            // );
            let mut rng = rand::thread_rng();

            if input_node_id == output_node_id
                || self
                    .inputs
                    .iter()
                    .any(|input| *input == output_node_id)
                || self
                    .outputs
                    .iter()
                    .any(|output| *output == input_node_id)
            {
                return; // loop or intra-extrema connection
            }
            let output_node = self.get_node(output_node_id);

            // TODO: verify this detects cycles
            // TODO: input intra-extrema connections not caught
            // walk routine: this should work for all cases: loop, cycle, output-input (extrema) edges in a graph
            //               also should prevent output->hidden connections leaving outputs with connection.len() == 0
            let mut next = output_node
                .read()
                .unwrap()
                .connections
                .iter()
                .map(|connection| connection.output_node.clone())
                .unique_by(|node| {
                    node.upgrade().unwrap().read().unwrap().id
                })
                // this is fine since connection is clone of Arc
                .collect::<Vec<Weak<RwLock<node>>>>();

            if next.par_iter().any(|node| {
                let id = node.upgrade().unwrap().read().unwrap().id;
                // TODO: should only need to look for input node
                (id == input_node_id) || (id == output_node_id)
            }) {
                // NOTE: the proposed connection will create a cycle so we silently ignore
                // println!(
                //     "FAILED TO ADD CONNECTION {}->{} is a cycle",
                //     input_node_id, output_node_id
                // );
                return;
            }
            // either we reach the output vector or we find the output_node
            while next.len() != 0 {
                next = next
                    .into_iter()
                    .unique_by(|node| {
                        node.upgrade().unwrap().read().unwrap().id
                    })
                    .par_bridge()
                    .map(|node| {
                        node.upgrade()
                            .unwrap()
                            .read()
                            .unwrap()
                            .connections
                            .iter()
                            .map(|edge| edge.output_node.clone())
                            .collect::<Vec<Weak<RwLock<node>>>>()
                    })
                    .flatten()
                    .collect::<Vec<Weak<RwLock<node>>>>();

                if next.par_iter().any(|node| {
                    let id =
                        node.upgrade().unwrap().read().unwrap().id;
                    (id == input_node_id) || (id == output_node_id)
                }) {
                    // NOTE: the proposed connection will create a cycle so we silently ignore
                    // println!(
                    //     "FAILED TO ADD CONNECTION {}->{} is a cycle",
                    //     input_node_id, output_node_id
                    // );
                    return;
                }
            }

            // add the connection to the output node
            let mut output_node = self.get_node(output_node_id);
            let new_connection = Arc::new(connection {
                output_node: Arc::downgrade(&output_node),
                param: rng.gen::<u8>(),
            });
            output_node
                .write()
                .unwrap()
                .input_connections
                .push(Arc::downgrade(&new_connection));
            // add the connection to the input node
            let mut input_node = self.get_node(input_node_id);
            input_node
                .write()
                .unwrap()
                .connections
                .push(new_connection);
        }
        /// split an existing connection to add a node to
        /// the network takes a node index and a
        /// connection index belonging to the
        /// node. initializes connections with random
        /// parameters.
        pub fn add_node(
            &mut self,
            node_index: usize,
            connection_index: usize,
            cur_node_id: usize,
        ) {
            let mut rng = rand::thread_rng();

            let mut new_node = Arc::new(RwLock::new(node {
                id: cur_node_id,
                connections: vec![],
                input_connections: vec![],
            }));

            // get the connection's vertices from the graph
            let input_node = self.get_node(node_index);
            let output_node =
                input_node.read().unwrap().connections.clone()
                    [connection_index]
                    .output_node
                    .clone();

            // add the new_node's input connection to the
            // graph
            let new_input_connection = Arc::new(connection {
                output_node: Arc::downgrade(&new_node),
                param: rng.gen::<u8>(),
            });
            input_node
                .write()
                .unwrap()
                .connections
                .push(new_input_connection.clone());
            // add the new input_connection to new_node
            new_node
                .write()
                .unwrap()
                .input_connections
                .push(Arc::downgrade(&new_input_connection));
            // add the new_node's output connection to the
            // graph
            let new_output_connection = Arc::new(connection {
                output_node: output_node.clone(),
                param: rng.gen::<u8>(),
            });
            new_node
                .write()
                .unwrap()
                .connections
                .push(new_output_connection.clone());
            // add the new out_connection to output_node
            output_node
                .upgrade()
                .unwrap()
                .write()
                .unwrap()
                .input_connections
                .push(Arc::downgrade(&new_output_connection));

            self.tensor.push(new_node);
        }
        /// add a random node by splitting a connection in the network
        pub fn random_node(&mut self) {
            let mut rng = rand::thread_rng();
            // get a random connection to split
            let node_select =
                rng.gen_range(self.outputs.len()..self.tensor.len());
            let connection_select = rng.gen_range(
                0..self
                    .get_node(node_select)
                    .read()
                    .unwrap()
                    .connections
                    .len(),
            );
            self.add_node(
                node_select,
                connection_select,
                self.tensor.len(),
            );
        }
        // TODO: dont use num_inputs here use self.inputs
        /// attempt to add a random connection to the network
        pub fn random_connection(&mut self, num_inputs: usize) {
            let mut rng = rand::thread_rng();

            // TODO: this causes false short circuiting of distribution
            let mut second_node_select =
                rng.gen_range(0..self.tensor.len());
            while (self.outputs.len()..num_inputs)
                .into_iter()
                .any(|input| second_node_select == input)
            {
                second_node_select =
                    rng.gen_range(0..self.tensor.len());
            }

            let mut first_node_select = 0; // garunteed to be output node
                                           // hillbilly replacement distribution fix
            while (first_node_select == first_node_select)
                && first_node_select == 0
            {
                first_node_select = rng
                    .gen_range(self.outputs.len()..self.tensor.len());
            }
            self.add_connection(
                first_node_select,
                second_node_select,
            );
        }

        // TODO: split_tree_depths to get innovation
        //       and compare different topologies.

        // TODO: rework this to a sorting routine that does
        // not need to cycle the network as an
        // infinite iterator while loop
        // TODO: recurrent connections *with determinism*

        // TODO: rewrite for simpler input_connection based forward propagation
        // NOTE: this technically can allow parallel edges.
        //       with recurrence this will allow all graphs to be propagates.
        /// forward propagate the given signals through the
        /// network and return the output node
        /// values. signals are signals arriving at each
        /// input node there should be one signal
        /// per input node.
        pub fn forward_propagate(&self, signals: Vec<u8>) -> Vec<u8> {
            // TODO: assert vector
            // dimensions are appropriate for this
            // network: self.input_nodes.len == signals.len
            // TODO: remove .clone, .cloned and .collect
            //
            // Since this is a nodal representation of neural network
            // data structure the buffer is associated signals with
            // nodes (ready to broadcast)
            let initialization = self
                .tensor
                .par_iter()
                .skip(self.outputs.len())
                .take(signals.len())
                .zip(signals)
                .collect::<Vec<(&Arc<RwLock<node>>, u8)>>();
            // println!(
            //     "FORWARD_PROP initialized: {:?}\n",
            //     initialization
            // );
            //
            // TODO: rework to have buffer entries be (nodeid, Vec<input_connections, signals>)
            //       stop when all nodeids are output nodes and perform one last sum outside
            //       loop then return. dont overthink it.
            //
            // perform the initial broadcast (without activation)
            let mut buffer =
                initialization
                    .into_iter()
                    .map(|node| {
                        node.0
                            .read()
                            .unwrap()
                            .connections
                            .clone()
                            .into_iter()
                            .map(move |connection_param| {
                                (
                                    self.get_node(
                                        connection_param
                                            .output_node
                                            .upgrade()
                                            .unwrap()
                                            .read()
                                            .unwrap()
                                            .id,
                                    ),
                                    vec![(
                                        connection_param.clone(),
                                        weights::weight(
                                            node.1,
                                            connection_param.param,
                                        ),
                                    )],
                                )
                            })
                    })
                    .flatten()
                    // TODO: this is complicated enough to justify a
                    //       data structure or anonymous data structure instead of tuple
                    .collect::<Vec<(
                        Arc<RwLock<node>>,
                        Vec<(Arc<connection>, u8)>,
                    )>>();
            // println!("initial broadcast complete:{:?}", buffer);
            //
            // TODO: integrate this and above into one bootstrap step
            // resort into node groups
            buffer =
                buffer
                    .into_iter()
                    .group_by(|node| node.0.read().unwrap().id)
                    .into_iter()
                    .map(|(key, group)| {
                        // collapse into one node
                        (
                        self.get_node(key),
                        group
                            .into_iter()
                            .map(|node| node.1.clone())
                            .flatten()
                            .collect::<Vec<(Arc<connection>, u8)>>(),
                    )
                    })
                    .collect::<Vec<(
                        Arc<RwLock<node>>,
                        Vec<(Arc<connection>, u8)>,
                    )>>();
            // println!(
            //     "FORWARD_PROP sorted with buffer: {:?}\n",
            //     buffer
            // );
            //
            // performs sum-normalize, activation, and next_node weighting per iteration
            // to maximize operations per address indirection and ready_node lookup
            while !buffer.par_iter().all(|node| {
                self.outputs.par_iter().any(|output| {
                    node.0.read().unwrap().id == *output
                })
            }) {
                // TODO: remove collections and clones in this part of the loop
                //       iterate all network parameters by reference here
                // NEED REFERENCES DUE TO POINTER COMPARISON IN READY_NODE

                println!(
                    "NODES IN BUFFER: {:?}\n",
                    buffer
                        .iter()
                        .map(|node| node.0.read().unwrap().id)
                        .collect::<Vec<usize>>()
                );

                // NOTE: this is actually really good for hoisting from for loop
                //       iff vec macro initializes allocation here
                let ready_nodes = RwLock::new(Vec::new());
                let halted_nodes = RwLock::new(Vec::new());
                //
                // check if nodes are ready to propagate
                // short-circuiting node-local search
                buffer.into_par_iter().for_each(|node| {
                    if node
                        .0
                        .read()
                        .unwrap()
                        .input_connections
                        .par_iter()
                        // TODO: false positive if no input_connections
                        .all(|input_connection| {
                            node.1.par_iter().any(
                                |buffer_connection| {
                                    Arc::ptr_eq(
                                        &input_connection
                                            .upgrade()
                                            .unwrap(),
                                        &buffer_connection.0,
                                    )
                                },
                            )
                        })
                        // TODO: can also be cmp to self.outputs
                        && node.0.read().unwrap().connections.len()
                            > 0
                    {
                        ready_nodes
                            .write()
                            .unwrap()
                            .push(node.clone());
                    } else {
                        halted_nodes
                            .write()
                            .unwrap()
                            .push(node.clone());
                    }
                });

                let mut ready_nodes =
                    ready_nodes.into_inner().unwrap();
                let mut halted_nodes =
                    halted_nodes.into_inner().unwrap();
                //
                // println!("FORWARD PROPAGATING WITH READY NODES {:?} AND HALTED NODES {:?}", ready_nodes, halted_nodes);
                //
                // propagate ready nodes
                let mut ready_nodes = ready_nodes.into_par_iter()
                    // .inspect(|activation| println!("activating: {:?}",activation))
                    .map(|node| {
                        // get the broadcast signal from this node
                        let broadcast_signal =
                        // @DEPRECATED: or justify
                        //activations::cond_rot_act(
                        // NOTE: this is not parallelized here due to associative property?
                        node.1.into_par_iter()
                        .map(|connection_param| {
                            weights::weight(connection_param.1, connection_param.0.param)
                        })
                        .reduce_with(|a: u8,b: u8|{
                            (a + b) >> 1
                        });
                        node.0.read().unwrap().connections.par_iter().map(|out_connection|{
                            (out_connection.output_node.upgrade().unwrap().clone(), vec![(out_connection.clone(), broadcast_signal.unwrap())])
                        }).collect::<Vec<(Arc<RwLock<node>>, Vec<(Arc<connection>, u8)>)>>()
                    })
                    .flatten()
                    // add halted nodes
                    .chain(halted_nodes.into_par_iter())
                    .collect::<Vec<(Arc<RwLock<node>>, Vec<(Arc<connection>, u8)>)>>();

                ready_nodes.par_sort_by(|node_a, node_b| {
                    Ord::cmp(
                        &node_a.0.read().unwrap().id,
                        &node_b.0.read().unwrap().id,
                    )
                });
                //
                // TODO: all of this should be parallel? this may be really fast anyways
                //      would like to remove clone here so may rework
                buffer =
                    ready_nodes
                        .into_iter()
                        .group_by(|node| node.0.read().unwrap().id)
                        .into_iter()
                        .map(|(key, group)| {
                            (
                                // TODO: should have to fetch this..
                                self.get_node(key),
                                group
                                    .into_iter()
                                    .map(|group_node| {
                                        group_node.1.clone()
                                    })
                                    .flatten()
                                    .collect::<Vec<(Arc<connection>, u8)>>(
                                    ),
                            )
                        })
                        .collect::<Vec<(
                            Arc<RwLock<node>>,
                            Vec<(Arc<connection>, u8)>,
                        )>>();
            }

            // DONE WITH HIDDEN LAYERS
            // now sum-normalize and activate the output vector
            buffer.par_sort_by(|node_a, node_b| {
                Ord::cmp(
                    &node_a.0.read().unwrap().id,
                    &node_b.0.read().unwrap().id,
                )
            });
            buffer
                .iter()
                .group_by(|node| node.0.read().unwrap().id)
                .into_iter()
                .map(|(_key, group)| {
                    group.into_iter().collect::<Vec<&(
                        Arc<RwLock<node>>,
                        Vec<(Arc<connection>, u8)>,
                    )>>()
                })
                .collect::<Vec<
                    Vec<&(
                        Arc<RwLock<node>>,
                        Vec<(Arc<connection>, u8)>,
                    )>,
                >>()
                .into_par_iter()
                .map(|group| {
                    //activations::cond_rot_act(
                    group
                        .into_iter()
                        .map(|node| {
                            node.1.iter().map(|connection_signal| {
                                connection_signal.1
                            })
                        })
                        .flatten()
                        .par_bridge()
                        .reduce_with(|a: u8, b: u8| (a + b) >> 1)
                        .unwrap()
                })
                .collect::<Vec<u8>>()
        }

        /// Calculates the gradient from an output vector and the true_output vector
        /// updating the parameters.
        /// Topology is augmented with point mutations (multiple mutations possible per backprop)
        /// by setting a distribution given the higher order derivative of the loss function.
        /// Point mutations means at scale the network can still complexify efficiently given samples.
        ///     connection_weight: update with normal backpropagation (directly).
        ///     add_connection: if error is decreasing or zero, increase connection mutation
        ///                      rate for this connection's input_node to a nearby
        ///                      (in split_node_tree) low performing node.
        ///                      (increase this signals usage in nearby approximators)
        ///                      distance of output_node in split_node_tree increases with mutation rate.
        ///                      s.t. zero error can connect anywhere in the topology.
        ///     add_node: if error is increasing or max, increase node mutation rate
        ///                for this connection.
        ///                 (add terms to find a better approximation/fit)
        ///
        /// this is loosely based on series approximation using polynomials or linear/sigmoidal
        /// functions where a best fit is incremented atomically with additional rank/terms.
        pub fn augmenting_gradient_descent(
            &self,
            outputs: Vec<u8>,
            true_outputs: Vec<u8>,
        ) {
            // NOTE: with ancestral speciation this should be performant without batching.
            // calculate the error
            let initial_differential = true_outputs
                .into_iter()
                .zip(outputs)
                .map(|error| {
                    // we check distance without pyth. thm.
                    let mut error_distance = 0;
                    if error.0 > error.1 {
                        error_distance = error.0 - error.1;
                    } else {
                        error_distance = error.1 - error.0;
                    }
                    error_distance
                })
                .collect::<Vec<u8>>();

            //begin the backprop through parameter space
            // TODO: need input_connection addresses in Node
        }
    }

    // sort the network nodes so forward_propagation can occur
    // in one cycle by forward_propagating and pushing based on
    // activation iteration.
    // pub fn pre_sort()

    // TRAITS //
    // NOTE: using iterators one can implement a custom
    // counting type to address larger than
    // architecture's usize such as 127 bit type but
    // should be considered in conjunction with
    // MMX/AVX passes in LLVM. This kind of free runtime
    // feature is awesome! I want to be a part of this.
    //
    // TODO: DoubleEndedIterator for backprop
    // TODO: REIMPLEMENT? Is this useful?
    //impl Iterator for rot_net {
    //    type Item = Vec<Box<[u8]>>;

    //    fn next(&mut self) -> Option<Vec<Box<[u8]>>> {
    //    }
    //}

    // used for debuging
    ///prints out the connections that define this rot_net
    impl fmt::Display for rot_net {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            let mut printout = "\n".to_string();
            let mut buffer = "".to_string();

            self.tensor.iter().for_each(|node| {
                buffer += &node.read().unwrap().id.to_string();
                buffer += &format!("->").to_string();
                buffer += "|";
                buffer += &node
                    .read()
                    .unwrap()
                    .connections
                    .iter()
                    .map(|connection| {
                        connection
                            .output_node
                            .upgrade()
                            .unwrap()
                            .read()
                            .unwrap()
                            .id
                            .to_string()
                            + "|"
                    })
                    .collect::<String>()
                    .to_string();
                buffer += &"\n".to_string();
                printout += &buffer.to_string();
                buffer.clear();
            });
            write!(f, "{}", printout)
        }
    }
}

//  TODO: initialize with limitation parameters as
// configuration consts.        force usize to be
// maximum connection per node and maximum network size in
// search.  TODO: PoM will also be limited by this but
// use agglomerative clustering to shrink the
//  RoM when some threshold of PoMs is reached. This allows
// continuous exploring of the RoM until
//  some equillibrium RoM representation is reached for a
// given search distance. varying  the search distance
// over time such as epsilon decay or proportional epsilon
// for PoM  count may be interesting. The goal is that
// this will be as robust as SARSA but with
//  some compression through learning. Knowledge based
// compression as well as mapping  the model and not
// the specific domain allows scale out to transfer learning
// across  domains and ranges (e.g.: sensors and
// fitness functions).
