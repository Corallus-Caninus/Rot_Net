// #![feature(cell_update)]
// NOTE: if you dont got a barrel shifter you probably shouldnt be doing data analytics..

// NOTE: this implementation breaks the borrow checker but
//      doesnt use Rc so can panic but should never leak.
//      Should only panic when parallelizing or asyncing improperly.

// TODO: clean up references and clones! make space efficient before
//       further speed optimization. unless some serious backend or
//       frontend optimization occurs I dont see how this doesnt bloat.
// TODO: Read source on move, copy and clone.
pub mod rot_net {
    use activations::{cond_rot_act, cond_rot_grad};
    use itertools::Itertools;
    // TODO: rng needs to be all passed in or called by reference.
    //       need to ensure not creating too many PRNG states (what is thread_local rng)
    use rand::Rng;
    use rand::{random, seq::IteratorRandom};
    use std::{
        borrow::BorrowMut,
        cell::{Cell, RefCell},
    };

    //TODO: use generics to ingest any data type and automatically vectorize larger precision types/classes into self.inputs
    //      cast to bits and discretize/bin? want to be able to send any object and chunk/vectorize into a network.
    //      std::mem::transmute might be a good option here. some methods within serde might also be helpful.
    //     use same generic casting schema for the output harvesting.
    //TODO: can we center all data points on ingest to 147 and implement a psuedo 0 crossing sigmoid
    //TODO: macro generic?
    //pub mod vectorize(){
    //  pub fn inputs<T>(Vec<T>)-> Vec<u8>
    //  pub fn outputs<T>(Vec<T>)->Vec<u8>}

    pub mod activations {
        //      hard code a limit to max connection in architecture search or implementation.
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
        //       weight space (mutate piecewise without LUT)? relu and other acti funcs show that its more
        //       than a weighted normalization and is the term of approximation. start with what works and
        //       investigate empirically.
        // TODO: shift the head over into the domain and set slopes as 2 and 4 respectively.
        /// approximate a sigmoid inflection and concavity features with a rotate of u8's.
        /// This implementation requires normalization of the connection parameters to prevent overflow.
        pub fn cond_rot_act(x: u8) -> u8 {
            const SEGMENTS: [u8; 5] = [30, 75, 128, 170, 255];
            //SEGMENTATION DESCRIPTION:
            // 1. tail of sigmoid is somewhat arbitrarily set @ 30 which is 10% of domain.
            // 2. head of sigmoid is clipped at 255, NOTE: sigmoid spans parameter precision
            //    not cur_buffer precision, this may change.
            // The lines and intercepts are solved for two slopes: 1/2 and 2 given 1. and 2.
            // This approximation is the maximum precision and minimum error optimizing for speed

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

    #[derive(Clone, Copy)]
    pub struct Connection<'a> {
        pub out_node: &'a Node<'a>,
        // pub in_node: &'a Node<'a>,
        // TODO: since these are pointed to from nodes we
        //       can keep adding binary resolution to condition trees
        //       would conditional bitmasking be faster given word size?
        //
        // TODO: byte this out dont use booleans each is a byte of bloat.
        // TODO: bitmask this piece.. what is the space-time tradeoff here?
        // NOTE: can increase statics to have 9! conditions only scales static comparators in weight
        //       this allows for all sorts of function approximations.
        // 0b1000000 => direction * or /
        // 0b0100000 => one shift * or / 2
        // 0b0010000 => two shift *or / 4
        //
        // TODO: may need to cell here so immutable with multiple node references (in and out)
        pub params: u8, // this is up to 8 conditions per const BITCHECK make 'em count.
    }
    impl<'a> Connection<'a> {
        pub fn weight(&self, sig: u8) -> u8 {
            // I like the depth of these conditions as apposed to unrolled elif elif etc.
            // conditional branching is faster just like a tree is faster than a list
            // and *should* have better branch prediction given posterior indirection.
            // TODO: functionally declare this with prototyped levels of conditions and rotations
            //       using traits if such resolution shows promise: impl weight for Connection{}
            //       declare this in psyclone/implementation and virtualize here?
            //  IMPLEMENTATIONS:
            //  1. notch_rot // is slow. has alot of expressivity but there are better ways to bit approx functions.
            //  2. n resolution (sig <</>> n)
            // TODO: can this be differentiated by retaining signals in parameters?
            //       less parameter-space efficient but just as fast forwards and backwards
            // TODO: should this return a 16? can get more domain out of rot but I designed this as u8
            //       so requires some further research

            // TODO: unittest this.

            // notch out a bitflag compare. currently 2 ops and a cmp (3 ops) can XOR bitmasking make this faster?
            // can 11111111 && self.params ^ 11111011 == 0b00000100 but this is still 3 ops and scaling statics
            // once
            const BITCHECK: u8 = 0b10000000;
            //single rotation
            if self.params >> 1 << 7 == BITCHECK {
                // direction bitflag is LSB so we just wipe other bits
                // will get optimized further in the backend for conditions
                // so this is enough..
                if self.params << 7 == BITCHECK {
                    sig >> 1
                } else {
                    sig << 1
                }
            //double rotation
            } else if self.params >> 2 << 7 == BITCHECK {
                if self.params << 7 == BITCHECK {
                    sig >> 2
                } else {
                    sig << 2
                }
            } else {
                sig //pass the signal fallthrough
            }
        }

        // TODO: UNUSED/DEPRECATED
        /// weight this signal and return it with the edge pointed Node
        pub fn activate(&self, sig: u8) -> (&Node<'a>, u8) {
            let res = self.weight(sig);
            (&self.out_node, res)
        }
    }
    #[derive(Clone)]
    pub struct Node<'a> {
        // NOTE: RefCell is prevented in share and send but this should be
        //       data parallel in group operations.
        //      just own everything in rot_net and reference? does this incur indirection?
        //      owning connections here will copy in forward_prop. resolve this or just own
        //      in rot_net and point anyways. Copy or Reference? (read rust)
        // TODO: does RefCell<Vec> make Connections mutable? this is too much overhead.
        pub out_edges: RefCell<Vec<&'a Cell<Connection<'a>>>>,
        pub in_edges: RefCell<Vec<&'a Cell<Connection<'a>>>>,
    }
    // TODO: here and rot_net may point to different objects due to cloneing.
    //       normally id say dont compare pointers but thats a workaround for bloat-safety.
    impl<'a> Node<'a> {
        // TODO: @DEPRECATED
        // add a connection between from this node to other.
        // pub fn add_connection(self, edge: &'a Cell<Connection<'a>>) {
        //     // edge.get().out_node.in_edges.borrow_mut().push(edge);
        //     // self.out_edges.borrow_mut().push(edge);
        //     self.out_edges.borrow_mut().push(edge);
        //     edge.get()
        //         .out_node
        //         .in_edges
        //         .borrow_mut()
        //         .push(edge);
        //     // TODO: shrink_to_fit each time if fragmented vecs cause capacity bloat
        // }
        // // // connect this node to the given node.
        // // // TODO: this cant be mutable.. nodes are immutably pointed to!!
        // pub fn add_node(self, out_node: &'a Node<'a>, new_connection: &'a Cell<Connection<'a>>) {
        //     // TODO: since this copies do we bloat memory in parameters?

        //     self.out_edges.borrow_mut().push(new_connection);
        //     out_node
        //         .in_edges
        //         .borrow_mut()
        //         .push(new_connection);
        //     // TODO: simplified test results show this is fine but verify that these are truly
        //     //       pointing to the correct places.

        //     // TODO: shrink_to_fit each time if fragmented vecs cause capacity bloat
        //     //       literally random so I doubt capacity will have any intuition in global allocator.
        //     //       may be useful if using gradient based architecture search..
        // }
    }
    #[derive(Clone)]
    pub struct network<'a> {
        // NOTE: if confused about this and construction here and
        //       in node/connection review ownership vs borrow-checker
        // nodes: Vec<Node<'a>>,
        // these are handles into and out of the network
        inputs: Vec<&'a Node<'a>>,
        outputs: Vec<&'a Node<'a>>,
    }
    impl<'a> network<'a> {
        // MUTABLE OPERATIONS //
        // INITIALIZATION ROUTINES //
        // TODO: because I am self taught rustacean, everything is passed between these methods and the calling frame.
        //       this is TODO: refactor to a single call and something more rust appropriate.
        pub fn initialize_nodes(
            num_inputs: i64,
            num_outputs: i64,
            // connections: Vec<&'a Cell<Connection<'a>>>,
        ) -> (Vec<Node<'a>>, Vec<Node<'a>>) {
            // TODO: add Connections
            let mut inputs = vec![];
            let mut outputs = vec![];
            for _i in 0..num_inputs {
                //create input node
                let next_node: Node<'a> = Node {
                    in_edges: RefCell::new(vec![]),
                    out_edges: RefCell::new(vec![]),
                };
                inputs.push(next_node);
            }
            for _i in 0..num_outputs {
                //create output node
                let next_node = Node {
                    in_edges: RefCell::new(vec![]),
                    out_edges: RefCell::new(vec![]),
                };
                outputs.push(next_node);
            }
            (inputs, outputs)
        }
        /// returns initial connections given initial output nodes.
        pub fn initialize_connections(
            ins: Vec<&'a Node<'a>>,
            outs: Vec<&'a Node<'a>>,
            rng: Vec<u8>,
        ) -> Vec<Cell<Connection<'a>>> {
            // TODO: pass in generic rand for weighting here.
            // TODO: not fully connected in current implementation!
            //      this needs to iter on inputs and map to outputs then reassociate in initializer.
            //      order doesnt matter because random params anyways just need set magnitude
            //      equivalence per output
            let mut res = vec![];
            for _j in 0..ins.len() {
                // TODO: this that rust move stuff i dont like.. why clone..
                for i in outs.iter().zip(rng.clone()) {
                    let cur = Connection {
                        out_node: *i.0,
                        params: i.1,
                    };
                    res.push(Cell::new(cur));
                }
            }
            res
        }
        /// create a fully connected topology with input and output
        /// nodes and randomized connections.
        pub fn rot_net_initialize(
            //TODO: if we create struct layer(Vec<&'a Node<'a>>) we
            //      can impl iterator for random walk etc.
            inputs: Vec<&'a Node<'a>>,
            outputs: Vec<&'a Node<'a>>,
            mut connections: Vec<&'a Cell<Connection<'a>>>,
        ) -> Self {
            let res = network {
                inputs: inputs,
                outputs: outputs,
            };
            // NOTE: these have to line up positionally until all constructor
            //       routines are consolidated into 1
            for input in res.inputs.iter() {
                for output in res.outputs.iter() {
                    //hillbilly X-Mas assertion
                    let cur_connection = connections.pop().unwrap();
                    input.out_edges.borrow_mut().push(cur_connection);
                    output.in_edges.borrow_mut().push(cur_connection);
                }
            }
            res
        }
        // COMPLEXIFYING ROUTINES //
        // TODO: how much of this should be extracted to vs how much should be called from Psyclones

        /// random walk this network for a Node
        /// rng should be no larger than the network otherwise the network will
        /// be walked repeatedly until rng is reached, not bad just sub-optimal and TODO.
        /// NOTE: this cannot connect input and output nodes (intra extrema connections)
        pub fn random_walk<R: Rng>(&self, rng: &'a mut R) -> &'a Node {
            let mut buffer = vec![];
            // TODO:  this walk should be an impl Iter for network
            //          iter().choose() but iters are strange..
            buffer = self
                .inputs
                .iter()
                .flat_map(|x| {
                    x.out_edges
                        .borrow()
                        .to_owned()
                        .into_iter()
                        .map(|y| y.get().out_node)
                })
                .collect();
            for _i in 0..rng.gen::<usize>() {
                buffer = buffer
                    .iter()
                    .flat_map(|x| {
                        x.out_edges
                            .borrow()
                            .to_owned()
                            .into_iter()
                            .map(|y| y.get().out_node)
                    })
                    .collect();
                if buffer.len() == 0 {
                    buffer = self
                        .inputs
                        .iter()
                        .flat_map(|x| {
                            x.out_edges
                                .borrow()
                                .to_owned()
                                .into_iter()
                                .map(|y| y.get().out_node)
                        })
                        .collect();
                }
            }
            buffer.iter().choose(rng).unwrap()
        }
        /// return a connection set between random nodes
        pub fn random_connection_nodes<R: Rng>(
            &'a self,
            first_rng: &'a mut R,
            second_rng: &'a mut R,
            third_rng: &'a mut R,
        ) -> (&Node, &Node) {
            // TODO: remove all these mutable references to an Rng..
            let first_node = self.random_walk(first_rng);
            let second_node = self.random_walk(second_rng);
            // TODO: from calling scope..
            // TODO: ensure this isnt an intra extrema connection. need
            //       to analyse topology to see if this is possible.
            let res = Cell::new(Connection {
                out_node: first_node,
                params: third_rng.gen(),
            });
            (first_node, second_node)
        }
        /// add a connection between two random nodes
        /// first and second rng should point to the same Rng object?
        /// NOTE: doesn't currently support connections between outputs and inputs or
        /// across extrema nodes (input->input and output->output)
        pub fn add_connection(
            self,
            new_connection: &'a Cell<Connection<'a>>,
            in_node: &'a Node<'a>,
            out_node: &'a Node<'a>,
        ) {
            in_node.out_edges.borrow_mut().push(new_connection);
            out_node.in_edges.borrow_mut().push(new_connection);
        }

        // TODO: this should be wrapped so a connection is the signature.
        //       may need to be written in implementing class
        /// add a node to this network by splitting a connection
        pub fn add_random_split_node<R: Rng>(
            self,
            in_node: &'a Node<'a>,
            in_connection: &'a Cell<Connection<'a>>,
            out_connection: &'a Cell<Connection<'a>>,
        ) {
            in_node.out_edges.borrow_mut().push(in_connection);
            in_node.out_edges.borrow_mut().push(out_connection);
        }

        // IMMUTABLE OPERATIONS //
        // TODO: at some point this will be matrix ops instead so keep in mind when optimizing.
        /// forward propagate.
        /// clones signals and returns the output vector.
        pub fn cycle(self, signals: Vec<u8>) -> Vec<u8> {
            // Initialize: get signals into the topology and ready the first layer
            //  1. each input node gets a signal and broadcasts
            let mut buffer = vec![];
            //initialize
            // NOTE: does not support intra-extrema connections
            //       (input->input output->input etc. except init topology input->output)
            buffer = self
                .inputs
                .iter()
                .flat_map(|input| {
                    input
                        .out_edges
                        .borrow()
                        .to_owned()
                        .into_iter()
                        .zip(signals.clone().into_iter())
                })
                .collect();
            // can still short circuit here if careful due to toplogy constraints
            // (no subset of self.outputs.in_edges will ever equate the set of buffer)
            // TODO: reduce collects to aggresively become lazy
            // TODO: reduce pointers to be more readable. should optimize away anyways in backend
            while !buffer.iter().any(|connection| {
                self.outputs.iter().any(|output| {
                    output
                        .in_edges
                        .borrow()
                        .iter()
                        .any(|final_edge| std::ptr::eq(*final_edge, connection.0))
                })
            }) {
                //iterate a layer of propagation, retaining nodes that dont have all signals yet.
                // NOTE: this is useful for
                //      1. matrix multiplication tensor generation.
                //      2. recurrent connections only require a small change to node retention condition
                //         and a carry-over signal parameter.
                buffer = buffer
                    .into_iter()
                    .sorted_by_key(|assoc_connection| {
                        assoc_connection.0.get().out_node as *const Node
                    })
                    .group_by(|assoc_connection| assoc_connection.0.get().out_node as *const Node)
                    .into_iter()
                    .flat_map(|(_key, node_assoc_connections)| {
                        let node_assoc_connections =
                            node_assoc_connections.collect::<Vec<(&Cell<Connection>, u8)>>();
                        if node_assoc_connections.iter().any(|assoc_connection| {
                            assoc_connection
                                .0
                                .get()
                                .out_node
                                .in_edges
                                .borrow()
                                .iter()
                                .any(|ready_connection| {
                                    std::ptr::eq(*ready_connection, assoc_connection.0)
                                })
                        }) {
                            // node normalization and sum routine
                            // TODO: could weight these as they arrive? may make more smooth computation.
                            let mut sig_sum: u16 = 0;
                            node_assoc_connections.iter().for_each(|assoc_connection| {
                                sig_sum +=
                                    assoc_connection.0.get().weight(assoc_connection.1) as u16;
                                sig_sum = sig_sum >> 1;
                            });
                            // node activation and broadcast routine
                            // TODO: retain groupings so resort isnt as costly per layer
                            node_assoc_connections
                                .into_iter()
                                .flat_map(|assoc_connection| {
                                    assoc_connection
                                        .0
                                        .get()
                                        .out_node
                                        .out_edges
                                        .borrow()
                                        // TODO: this is not preferable but kinda makes sense
                                        .to_owned()
                                        .into_iter()
                                        .map(|next_connection| {
                                            (
                                                next_connection,
                                                activations::cond_rot_act(sig_sum as u8),
                                            )
                                        })
                                })
                                .collect()
                        } else {
                            //node delay routine
                            node_assoc_connections
                        }
                    })
                    .collect();
            }

            // TODO: activate one more time should extract operation first
            return buffer.iter().map(|x| x.1).collect();
        }
    }
}
