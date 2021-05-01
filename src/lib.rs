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
    use std::sync::{Arc, Mutex};

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
        // TODO: Arc Mutex is slow and bloated.
        //       would prefer data parallel locality instead
        //       if this is only addressing based solution

        // TODO: get rid of this Arc and use data-parallelism
        // TODO: this should be weak but lives as long as self
        //       since nodes arent pruned
        output_node: Arc<Mutex<node>>,
        // the ID of this connection
        innovation: usize,
        param: u8,
    }

    /// a vertex in the network graph
    #[derive(Clone, Debug)]
    pub struct node {
        // the ID of this node
        // NOTE: would prefer for this to be an address of a connection vector but is too dynamic
        id: usize,
        // the connections going out from this node
        connections: Vec<connection>,
        // the innovation numbers of connections going into this node
        input_connections: Vec<usize>,
    }
    // impl Ord for node<'a> {
    //     fn cmp(&self, other: &Self) -> Ordering {
    //         self.id.cmp(&other.id)
    //     }
    // }
    // impl PartialOrd for node<'a> {
    //     fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
    //         Some(self.cmp(other))
    //     }
    // }
    // impl PartialEq for node<'a> {
    //     fn eq(&self, other: &Self) -> bool {
    //         self.id == other.id
    //     }
    // }
    // impl Eq for node<'a> {} //Implied by PartialEq

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
        pub tensor: Vec<Arc<Mutex<node>>>,
        // the output node id's
        pub outputs: Vec<usize>,
        // current max_innovation prevents having to
        // count each time a connection is added *for the low low cost of 1 usize!*
        pub innovation_counter: usize,
    }
    // METHODS //
    impl rot_net {
        pub fn initialize_network(
            inputs: usize,
            outputs: usize,
        ) -> Self {
            let mut rng = rand::thread_rng();
            let mut tensor = vec![];
            let mut innovation_counter = 0;

            // add the output connections one time
            // TODO: rework this order of operations for
            // readability
            let mut output_vector = vec![];
            for output in 0..outputs {
                let mut new_node = node {
                    id: output,
                    connections: vec![],
                    input_connections: vec![],
                };
                // TODO: ensure this aligns correctly with input node creation
                // one input connection innovation each
                for input in 0..inputs {
                    innovation_counter += 1;
                    new_node
                        .input_connections
                        .push(innovation_counter);
                }
                tensor.push(Arc::new(Mutex::new(new_node)));
                output_vector.push(output);
            }

            // reset the counter
            innovation_counter = 0;
            // since we are considering recurrent
            // connections as buffer roll over
            // (not param associated so doesnt bloat)
            // we allow output nodes to exist in the
            // tensor. This is +1 timestep recurrence.
            for input in 0..inputs {
                println!("initializing with id {}", outputs + input);
                // TODO: input nodes are wrong
                let mut new_node = node {
                    id: input + outputs,
                    connections: vec![],
                    input_connections: vec![],
                };
                for output in 0..outputs {
                    let new_connection = connection {
                        output_node: tensor[output].clone(),
                        innovation: innovation_counter,
                        param: rng.gen::<u8>(),
                    };
                    new_node.connections.push(new_connection);
                    println!(
                        "initializing connection for node with id {}",
                        tensor.len()
                    );
                    innovation_counter += 1;
                }
                tensor.push(Arc::new(Mutex::new(new_node)));
            }
            rot_net {
                tensor: tensor,
                outputs: output_vector,
                innovation_counter: innovation_counter,
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
        pub fn get_node(&self, index: usize) -> Arc<Mutex<node>> {
            // TODO: output nodes arent considered in this
            self.tensor
                .iter()
                .find(|node| node.lock().unwrap().id == index)
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
            println!(
                "adding connection..{} {}",
                input_node_id, output_node_id
            );
            let mut rng = rand::thread_rng();

            if input_node_id == output_node_id {
                return; // loop
            };

            let output_node = self.get_node(output_node_id);

            // TODO: verify this detects cycles
            // walk routine: this should work for all cases: loop, cycle, output-input (extrema) edges in a graph
            //               also should prevent output->hidden connections leaving outputs with connection.len() == 0
            let mut next = output_node
                .lock()
                .unwrap()
                .connections
                .iter()
                .map(|connection| connection.output_node.clone())
                .unique_by(|node| node.lock().unwrap().id)
                // this is fine since connection is clone of Arc
                .collect::<Vec<Arc<Mutex<node>>>>();

            if next.par_iter().any(|node| {
                let id = node.lock().unwrap().id;
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
                    .unique_by(|node| node.lock().unwrap().id)
                    .par_bridge()
                    .map(|node| {
                        node.lock()
                            .unwrap()
                            .connections
                            .iter()
                            .map(|edge| edge.output_node.clone())
                            .collect::<Vec<Arc<Mutex<node>>>>()
                    })
                    .flatten()
                    .collect::<Vec<Arc<Mutex<node>>>>();

                if next.par_iter().any(|node| {
                    let id = node.lock().unwrap().id;
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

            self.innovation_counter += 1;
            // add the connection to the output node
            let mut output_node = self.get_node(output_node_id);
            let new_connection = connection {
                output_node: output_node.clone(),
                innovation: self.innovation_counter,
                param: rng.gen::<u8>(),
            };
            output_node
                .lock()
                .unwrap()
                .input_connections
                .push(self.innovation_counter);
            // add the connection to the input node
            let mut input_node = self.get_node(input_node_id);
            input_node
                .lock()
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

            let mut new_node = Arc::new(Mutex::new(node {
                id: cur_node_id,
                connections: vec![],
                input_connections: vec![],
            }));
            // TODO: check if we should be using weak references in Arc here for leaking

            // get the connection's vertices from the graph
            let input_node = self.get_node(node_index);
            let output_node =
                input_node.lock().unwrap().connections.clone()
                    [connection_index]
                    .output_node
                    .clone();

            // add the new_node's input connection to the
            // graph
            self.innovation_counter += 1;
            let new_input = connection {
                output_node: new_node.clone(),
                innovation: self.innovation_counter,
                param: rng.gen::<u8>(),
            };
            input_node.lock().unwrap().connections.push(new_input);
            // add the new input_connection to new_node
            new_node
                .lock()
                .unwrap()
                .input_connections
                .push(self.innovation_counter);
            // add the new_node's output connection to the
            // graph
            self.innovation_counter += 1;
            let new_output = connection {
                output_node: output_node.clone(),
                innovation: self.innovation_counter,
                param: rng.gen::<u8>(),
            };
            // TODO: does this deadlock?
            new_node.lock().unwrap().connections.push(new_output);
            // add the new out_connection to output_node
            output_node
                .lock()
                .unwrap()
                .input_connections
                .push(self.innovation_counter);

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
                    .lock()
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
        // TODO: dont use num_inputs here
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
        // TODO: recurrent connections
        // TODO: use node groupings to reduce filtering
        // recalculating and sorting
        // TODO: unless sorting can be done its worth it to
        //       add input_node to connections (searching takes
        //       MUCH longer than memory is consumed, O(n)ish )

        // NOTE: this technically can allow parallel edges.
        //       with recurrence this will allow all graphs to be propagates.
        // forward propagate the given signals through the
        // network and return the output node
        // values. signals are signals arriving at each
        // input node there should be one signal
        // per input node.
        // pub fn forward_propagate(&self, signals: Vec<u8>) -> Vec<u8> {
        //     // TODO: assert vector
        //     // dimensions are appropriate for this
        //     // network: self.input_nodes.len == signals.len
        //     // TODO: remove .clone, .cloned and .collect

        //     // Since this is a nodal representation of neural network
        //     // data structure the buffer is associated signals with
        //     // nodes (ready to broadcast)
        //     let initialization = self
        //         .tensor
        //         .iter()
        //         .take(signals.len())
        //         .zip(signals)
        //         .collect::<Vec<(&RefCell<node>, u8)>>();

        //     //println!(
        //     //"FORWARD_PROP initialized: {:?}\n",
        //     //initialization
        //     //);

        //     // TODO: rework to have buffer entries be (nodeid, Vec<input_connections, signals>)
        //     //       stop when all nodeids are output nodes and perform one last sum outside
        //     //       loop then return. dont overthink it.

        //     // perform the initial broadcast (without activation)
        //     let mut buffer = initialization
        //         .iter()
        //         // TODO: this is supposed to be output_connection.output_node node node.0.id
        //         .map(|node| {
        //             //(node.0, vec![(node.0.id, node.1)])
        //             node.0.borrow().connections.iter().map(
        //                 move |connection_param| {
        //                     (
        //                         self.get_node(
        //                             connection_param.output_node.id,
        //                         ),
        //                         vec![(
        //                             connection_param,
        //                             weights::weight(node.1, connection_param.param),
        //                         )],
        //                     )
        //                 },
        //             )
        //         })
        //         .flatten()
        //         // TODO: this is complicated enough to justify a
        //         //       data structure or anonymous data structure instead of tuple
        //         .collect::<Vec<(RefCell<node>, Vec<(&connection, u8)>)>>();

        //     // TODO: integrate this and above into one bootstrap step
        //     // resort into node groups
        //     buffer = buffer
        //         .iter()
        //         .group_by(|node| node.0.borrow().id)
        //         .into_iter()
        //         .map(|(key, group)| {
        //             // collapse into one node
        //             (
        //                 self.get_node(key).borrow(),
        //                 group
        //                     .into_iter()
        //                     .map(|node| node.1.clone())
        //                     .flatten()
        //                     .collect::<Vec<(&connection, u8)>>(),
        //             )
        //         })
        //         .collect::<Vec<(&node, Vec<(&connection, u8)>)>>();

        //     //println!(
        //     //"FORWARD_PROP sorted with buffer: {:?}\n",
        //     //buffer
        //     //);

        //     // performs sum-normalize, activation, and next_node weighting per iteration
        //     // to maximize operations per address indirection and ready_node lookup
        //     while !buffer.iter().all(|node| {
        //         self.outputs.iter().any(|output| node.0.borrow().id == *output)
        //     }) {
        //         //println!("BUFFER: {:?}\n", buffer);

        //         // NOTE: this is actually really good for hoisting from for loop
        //         //       iff vec macro initializes allocation here

        //         // let mut ready_nodes = Mutex::new(Vec::new());
        //         // let mut halted_nodes = Mutex::new(Vec::new());
        //         let mut ready_nodes = Vec::new();
        //         let mut halted_nodes = Vec::new();

        //         // check if nodes are ready to propagate
        //         // TODO: rework this better now that ready is a local search
        //         //       in buffer not tensor
        //         // short-circuiting node-local search
        //         buffer.iter().for_each(|node|{
        //             //if self.node_ready_comparator(node.0.borrow(), &node.1) {
        //             if node.0.borrow().input_connections.iter().all(|input_connection|{
        //                 node.1.iter().any(|buffer_connection|{
        //                     *input_connection == buffer_connection.0.innovation
        //                 })
        //             }){
        //                 ready_nodes.push(node.to_owned());
        //             } else {
        //                 halted_nodes.push(node.to_owned());
        //             }
        //         });
        //         // let mut ready_nodes: Vec<(&node, Vec<(&connection, u8)>)> = ready_nodes.lock().unwrap().into();
        //         // let mut halted_nodes = halted_nodes.lock().unwrap().into();

        //         // println!("FORWARD PROPAGATING WITH READY NODES {:?} AND HALTED NODES {:?}", ready_nodes, halted_nodes);

        //         // propagate ready nodes
        //         ready_nodes = ready_nodes.iter()
        //             // .inspect(|activation| println!("activating: {:?}",activation))
        //             .map(|node| {
        //                 // get the broadcast signal from this node
        //                 let broadcast_signal =
        //                 // @DEPRECATED: or justify
        //                 //activations::cond_rot_act(
        //                 // NOTE: this is not parallelized here due to associative property
        //                 node.1.iter().fold(0,|res,acc|{
        //                     (res >> 1) + (weights::weight(acc.0.param,acc.1) >> 1)
        //                 });
        //                 //);
        //                 node.0.borrow().connections.iter().map(|out_connection|{
        //                     //(self.get_node(out_connection.output_node.id), vec![(out_connection, broadcast_signal)])
        //                     (out_connection.output_node, vec![(out_connection, broadcast_signal)])
        //                 }).collect::<Vec<(&RefCell<node>, Vec<(&connection, u8)>)>>()
        //             })
        //             .flatten()
        //             // add halted nodes
        //             .chain(halted_nodes.into_iter())
        //             .collect::<Vec<(&RefCell<node>, Vec<(&connection, u8)>)>>();
        //             ready_nodes.sort_by(|node_a, node_b| {
        //                 Ord::cmp(&node_a.0.borrow().id, &node_b.0.borrow().id)
        //             });

        //             // TODO: all of this should be parallel? this may be really fast anyways
        //             //      would like to remove clone here so may rework
        //             buffer = ready_nodes.iter().group_by(|node| node.0.borrow().id)
        //             .into_iter()
        //             .map(|(key, group)| {
        //                 (
        //                     self.get_node(key),
        //                     group
        //                         .into_iter()
        //                         .map(|group_node| {
        //                             group_node.1.clone()
        //                         })
        //                         .flatten()
        //                         .collect::<Vec<(&connection, u8)>>(),
        //                 )
        //             })
        //             .collect::<Vec<(&RefCell<node>, Vec<(&connection, u8)>)>>();
        //     }

        //     // now sum-normalize and activate the output vector
        //         buffer.sort_by(|node_a, node_b| {
        //             Ord::cmp(&node_a.0.borrow().id, &node_b.0.borrow().id)
        //         });
        //         buffer
        //         .iter()
        //         .group_by(|node| node.0.borrow().id)
        //         .into_iter()
        //         .map(|(_key,group)|
        //         group.into_iter().collect::<Vec<&(&RefCell<node>, Vec<(&connection,u8)>)>>())
        //         .collect::<Vec<Vec<&(&RefCell<node>,Vec<(&connection, u8)>)>>>()

        //         .into_iter()
        //         .map(|group| {
        //             //activations::cond_rot_act(
        //                 group
        //                     .into_iter()
        //                     .map(|node| {
        //                         node.1.iter().map(
        //                             |connection_signal| {
        //                                 connection_signal.1
        //                             },
        //                         )
        //                     })
        //                     .flatten()
        //                     // TODO: is this associative/parallelizable? me thinks no
        //                     // NOTE: this is not parallelized
        //                     .fold(0, |acc, res| {
        //                         (acc >> 1) + (res >> 1)
        //                     })
        //             //)
        //         })
        //         .collect()
        // }

        //@DEPRECATED
        // check if all connections for a
        // node exist in a (connection,signal) buffer
        //
        // cur_in_connections: current in connections in a buffer
        //                     including this nodes in connections
        // node: the node being compared.
        // pub fn node_ready_comparator(
        //     &self,
        //     node: &node,
        //     cur_in_connections: &Vec<(&connection, u8)>,
        // ) -> bool {
        //     // all in_connections for this node
        //     let in_connections = self.get_in_connections(node.id);

        //     let is_output =
        //         self.outputs.iter().any(|output| *output == node.id);

        //     (in_connections.len() == cur_in_connections.len())
        //         && !is_output
        // // }
    }

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
                buffer += &node.lock().unwrap().id.to_string();
                buffer += &format!("->").to_string();
                buffer += "|";
                buffer += &node
                    .lock()
                    .unwrap()
                    .connections
                    .iter()
                    .map(|connection| {
                        connection
                            .output_node
                            .lock()
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

// TODO:
// cycle through the nodes halting if not ready for
// activation pub fn forward_propagate(){}
//
// sort the network nodes so forward_propagation can occur
// in one cycle by forward_propagating and pushing based on
// activation iteration.
// pub fn pre_sort(){}
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
//      allowing informative based complexification
// (generalization/abstraction). NOTE: This doesnt
// solve the multi optimization problem of architecture
// search      in itself. This is more of a sample
// efficiency solution. multi-optimization      can be
// solved by directly changing the topology using the
// gradient      information for
//      connection weight and taking a higher order
// derivative with the      gradient for complexification
// (but still using a scalar      variation to prevent
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
// and conditionally perform mutation_connection
// distribution based on error*(variance d/dx)        node
// selection also calculated from error*(variance d/dx) of
// other        weights possibly connecting to
// high performing nodes (average
//        input_connection error). what does hebbian
// plasticity say about this? can use        locality
// with average error (within same subtree) for
// fire-together        wire-together (distance-fitness
// selection metric). If a connection is
// underperforming look to integrate terms.     5. sample
// and conditionally perform mutation_node distribution
// based on error*(1/variance d/dx)        directly on the
// connection in question. if a connection is performing
// well add        terms.
//  6. return
//  such that mutation_weight_rate >
// mutation_connection_rate > mutation_node_rate loop can be
// implemented as BFS that halts a subtree backprop when the
// gradient is not taken. prefered to traverse the entire
// tree for simplicity and thoroughness. Sometimes there is
// a chaotic nature to the system that can be changed from a
// shallower point in the network (why does the
// backprop gradient weigh more on the deeper connection
// weights. This cant possibly be condusive to
// generalization and hints at a chaotic system seeded from
// inputs) The sparsity of the chaotic search is what
// makes ANN robust to high dimension data and
// functions allowing for generalization (transfer learning)
// and architecture reuse across domains.

//TODO: trace split_depths tree for innovation assignment
// by iterating rot_net     and getting
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
// over time such as epsilon decay or proportional epsilon
// for PoM  count may be interesting. The goal is that
// this will be as robust as SARSA but with
//  some compression through learning. Knowledge based
// compression as well as mapping  the model and not
// the specific domain allows scale out to transfer learning
// across  domains and ranges (e.g.: sensors and
// fitness functions).
