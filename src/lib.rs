// #![feature(cell_update)]
// NOTE: if you dont got a barrel shifter you probably shouldnt be doing data analytics..

// NOTE: this implementation breaks the borrow checker but
//      doesnt use Rc so can panic but should never leak.
//      Should only panic when parallelizing or asyncing improperly.
//
// TODO: all downstream ownership?: PROFILE FOR THIS DONT SPECULATE
//       edge: Rc<Node>
//       node{out_edges = RefCell<Vec<Cell<Connection>>>
//            in_edges = RefCell<Vec<&Cell<Connection>>>}
//       rot_net{inputs: Vec<Node>}
//       ..but this doesnt work for recurrent connections (leak)..
//       can use primal node as Rc and all others as weak but breaks
//       pruning and crossover topologies. how is this mutated?

// TODO: clean up references and clones! make space efficient before
//       further speed optimization. unless some serious backend or
//       frontend optimization occurs I dont see how this doesnt bloat.

// TODO: usize is the limiting address factor. ensure there is some safety around overflowing
//       usize. This may be solved with edge count and parameter limitations in architecture search.
//       this shouldnt happen. let the stack/heap overflow and fix when *if* that happens.

pub mod rot_net {
    use activations::{cond_rot_act, cond_rot_grad};
    use itertools::Itertools;
    // TODO: rng needs to be all passed in or called by reference.
    //       need to ensure not creating too many PRNG states (what is thread_local rng)
    use rand::prelude::*;
    use rand::{random, seq::IteratorRandom};
    use std::{
        borrow::BorrowMut,
        cell::{Cell, RefCell},
    };

    //TODO: use generics to ingest any data type and automatically vectorize
    //      larger precision types/classes into self.inputs
    //      cast to bits and discretize/bin? want to be able to send any object and
    //      chunk/vectorize into a network.
    //      std::mem::transmute might be a good option here. some methods within serde
    //      might also be helpful.
    //     use same generic casting schema for the output harvesting.

    //TODO: macro generic? so variable length (would also need variable type?)
    //pub mod vectorize(){
    //  pub fn inputs<T>(Vec<T>)-> Vec<u8>
    //  pub fn outputs<T>(Vec<T>)->Vec<u8>}

    //TODO: should this be an embedded module?
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
        // TODO: shift the head over into the domain and set slopes as 2 and 4 respectively.
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
            // The lines and intercepts are solved for two slopes: 1/2 and 2 given 1.
            // and 2. This approximation is the maximum precision and minimum error
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

    #[derive(Clone, Copy)]
    pub struct Connection<'a> {
        pub out_node: &'a Node<'a>,
        // TODO: since these are pointed to from nodes we
        //       can keep adding binary resolution to condition trees
        //       would conditional bitmasking be faster given word size?
        //
        // TODO: byte this out dont use booleans each is a byte of bloat.
        // TODO: bitmask this piece.. what is the space-time tradeoff here?
        // NOTE: can increase statics to have 9! conditions only scales static
        //       comparators in weight this allows for all sorts of function
        //       approximations.
        // 0b1000000 => direction * or /
        // 0b0100000 => one shift * or / 2
        // 0b0010000 => two shift *or / 4
        //
        // TODO: may need to cell here so immutable with multiple node references
        //       (in and out)
        pub params: u8, // this is up to 8 conditions per const BITCHECK make 'em count.
    }
    impl<'a> Connection<'a> {
        //TODO: derive this

        /// weight the given signal by this connection's params using rotations.
        pub fn weight(&self, sig: u8) -> u8 {
            // I like the depth of these conditions as apposed to unrolled elif elif etc.
            // conditional branching is faster just like a tree is faster than a list
            // and *should* have better branch prediction given posterior indirection.
            // TODO: functionally declare this with prototyped levels of conditions
            //       and rotations
            //       using traits if such resolution shows promise: impl weight for
            //       Connection{}
            //       declare this in psyclone/implementation and virtualize here?
            //  IMPLEMENTATIONS:
            //  1. notch_rot // is slow. has alot of expressivity but there are better
            //     ways to bit approx functions.
            //  2. n resolution (sig <</>> n)
            // TODO: can this be differentiated by retaining signals in parameters?
            //       less parameter-space efficient but just as fast forwards and backwards
            // TODO: should this return a 16? can get more domain out of rot but I
            //       designed this as u8
            //       so requires some further research

            // TODO: unittest this.

            // notch out a bitflag compare. currently 2 ops and a cmp (3 ops) can XOR
            // bitmasking make this faster?
            // can 11111111 && self.params ^ 11111011 == 0b00000100 but this is still 3
            // ops and scaling statics once.
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
    }
    #[derive(Clone)]
    pub struct Node<'a> {
        // NOTE: RefCell is prevented in share and send but this should be
        //       data parallel in group operations.
        pub out_edges: RefCell<Vec<&'a Cell<Connection<'a>>>>,
        pub in_edges: RefCell<Vec<&'a Cell<Connection<'a>>>>,
    }
    impl<'a> Node<'a> {}
    #[derive(Clone)]
    pub struct network<'a> {
        //TODO: if we create struct layer(Vec<&'a Node<'a>>) we
        //      can impl iterator for random walk etc.
        //      is it at all possible to impl iter for Vec<T>?
        pub inputs: Vec<&'a Node<'a>>,
        pub outputs: Vec<&'a Node<'a>>,
        //TODO: own here.. +1 indirection but its a start. matrix will probably solve this just
        //      ensure GA search isnt slow due to this. Anymore optimizations need to be profile
        //      justified.
        pub hidden_nodes: Vec<Node<'a>>,
        pub connections: Vec<Cell<Connection<'a>>>,
    }
    impl<'a> network<'a> {
        // MUTABLE OPERATIONS //
        // TODO: ensure these are threaded appropriately, various borrow_muts which might work in
        //       data parallel operations but will throw Send/Sync.
        // INITIALIZATION ROUTINES //
        // TODO: consolidate these into one method?
        // TODO: all construction methods should "generate" a value from the thread local rng..
        //       ensure not spinning up a PRNG each call/independantly in a thread stack.
        //       TEST AND CLOSE
        pub fn initialize_nodes(num_inputs: usize, num_outputs: usize) -> Vec<Node<'a>> {
            let mut owned_nodes = vec![];
            for _i in 0..num_inputs {
                //create input node
                let next_node: Node = Node {
                    in_edges: RefCell::new(vec![]),
                    out_edges: RefCell::new(vec![]),
                };
                owned_nodes.push(next_node);
            }
            for _i in 0..num_outputs {
                //create output node
                let next_node = Node {
                    in_edges: RefCell::new(vec![]),
                    out_edges: RefCell::new(vec![]),
                };
                owned_nodes.push(next_node);
            }
            owned_nodes
        }
        pub fn initialize_extrema<'b>(
            // TODO: this borrow needs to fall off?
            nodes: &'a Vec<Node<'a>>,
            num_inputs: usize,
            num_outputs: usize,
        ) -> (Vec<&'a Node<'a>>, Vec<&'a Node<'a>>) {
            use crate::rot_net::*;
            let inputs: Vec<&'a Node<'a>> = vec![];
            // TODO: cannot outlive borrowed content.. vec can change reference so these are locked
            // together
            let inputs = nodes.iter().take(num_inputs).collect::<Vec<&'a Node<'a>>>();
            let outputs = nodes
                .iter()
                .skip(num_inputs)
                .take(num_outputs)
                .collect::<Vec<&'a Node<'a>>>();
            (inputs, outputs)
        }
        /// returns initial connections given nodes.
        pub fn initialize_connections(
            inputs: &Vec<&'a Node<'a>>,
            outputs: &Vec<&'a Node<'a>>,
        ) -> Vec<Cell<Connection<'a>>> {
            let mut rng = rand::thread_rng();
            //TODO: need to add to input and output node edges
            println!("in initialize connections!");
            let mut res = vec![];
            for _j in 0..inputs.len() {
                for i in outputs.iter() {
                    let cur = Connection {
                        out_node: *i,
                        params: rng.gen(),
                    };
                    res.push(Cell::new(cur));
                }
            }
            res
        }
        pub fn initialize_network(&'a self) {
            // TODO: this is another place where elision or explicit lifetimes needs to drop borrow
            //       in call frame.
            // TODO: own connections in out_edges?
            self.inputs
                .iter()
                .zip(self.outputs.iter())
                .zip(self.connections.iter())
                .for_each(|connect| {
                    connect.0 .0.out_edges.borrow_mut().push(connect.1);
                    connect.0 .1.in_edges.borrow_mut().push(connect.1);
                });
        }

        // COMPLEXIFYING ROUTINES //
        // TODO: how much of this should be extracted to vs how much should be
        //       called from Psyclones

        // TODO: this cannot connect input and output nodes (intra extrema connections)
        /// random walk this network for a Node
        /// rng should not span larger than the network otherwise the network will
        /// be walked repeatedly until rng is reached, not bad just sub-optimal and TODO.
        pub fn random_walk(&'a self) -> &'a Node<'a> {
            let mut buffer = vec![];
            // TODO:  this walk should be an impl Iter for network
            //          iter().choose() requires: impl iter for layer{}
            //  TODO: can use cycle() here
            let mut rng = rand::thread_rng();

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
            buffer.iter().choose(&mut rng).unwrap()
        }

        /// return a connection set between random nodes.
        pub fn random_connection_nodes(
            &'a self,
        ) -> (&'a Node<'a>, &'a Node<'a>, Cell<Connection<'a>>) {
            let mut rng = rand::thread_rng();

            // get extrema nodes from network
            // TODO: call local self.local_extrema_nodes

            let first_node = self.random_walk();
            let second_node = self.random_walk();

            // TODO: deterministically analyse the network timeout is sloppy and can theoretically clip.
            let mut timeout = 0;
            const MAX_ATTEMPTS: i32 = 1000;
            // prevent extrema connections:
            // outputs to input connections
            while (self.outputs.iter().any(|x| std::ptr::eq(*x, first_node))
                && self.inputs.iter().any(|x| std::ptr::eq(*x, second_node)))
                //outputs to outputs connections
                || (self.outputs.iter().any(|x| std::ptr::eq(*x, first_node))
                    && self.outputs.iter().any(|x| std::ptr::eq(*x, second_node)))
                //input to input connections
                || (self.inputs.iter().any(|x| std::ptr::eq(*x, first_node))
                    && self.inputs.iter().any(|x| std::ptr::eq(*x, second_node)))
            {
                // retry if the network's connections arent depleted (marginally extrema-node fully
                // connected)
                timeout += timeout;
                if timeout > MAX_ATTEMPTS {
                    break;
                }
            }

            let res = Cell::new(Connection {
                out_node: first_node,
                params: rng.gen::<u8>(),
            });
            (first_node, second_node, res)
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
            // TODO: Remove clone here
            out_node.in_edges.borrow_mut().push(new_connection);
        }

        // TODO: this should be wrapped so a connection is the signature.
        //       may need to be written in implementing class
        /// add a node to this network by splitting a connection
        pub fn add_split_node(
            self,
            in_node: &'a Node<'a>,
            in_connection: &'a Cell<Connection<'a>>,
            out_connection: &'a Cell<Connection<'a>>,
        ) {
            in_node.out_edges.borrow_mut().push(in_connection);
            in_node.in_edges.borrow_mut().push(out_connection);
        }

        // IMMUTABLE OPERATIONS //
        /// Activate this node by summing, normalizing and activating this connection then
        /// broadcasting the result to the nodes out_edges and finally returning the
        /// associated connection-signal tuples.
        pub fn activate_nodes(
            node_assoc_connections: Vec<(&Cell<Connection<'a>>, u8)>,
        ) -> Vec<(&'a Cell<Connection<'a>>, u8)> {
            // TODO: could weight these as they arrive? may make more smooth computation.
            let mut sig_sum: u16 = 0;
            node_assoc_connections.iter().for_each(|assoc_connection| {
                sig_sum += assoc_connection.0.get().weight(assoc_connection.1) as u16;
                sig_sum = sig_sum >> 1;
            });
            // node activation and broadcast routine
            // TODO: retain groupings so resort isnt as costly per layer with retained
            // nodes.
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
                            (next_connection, activations::cond_rot_act(sig_sum as u8))
                        })
                })
                .collect()
        }

        // TODO: at some point this will be matrix ops instead so keep in mind when optimizing.
        // TODO: optimizations can occur in reducing collects and references. The later may get
        //       backend optimized out anyways with inlining and argpromotion.
        //       (does lazy analysis make algorithms lazy with
        //       inlining?)
        // TODO: this can be done with cycle instead of recursion. add a break condition and treat
        //       as while loop
        /// forward propagate this rot_net.
        /// clones signals and returns the output vector.
        pub fn cycle(&'a self, signals: Vec<u8>) -> Vec<u8> {
            // TODO: call self.local_extrema_nodes

            // Initialize: get signals into the topology and ready the first layer
            //  1. each input node gets a signal and broadcasts
            // TODO: filter out repeated connections
            let mut buffer = vec![];
            //initialize
            // NOTE: does not support intra-extrema connections
            //       (input->input output->input etc. except init topology input->output)
            buffer = self
                .inputs
                .iter()
                // TODO: inputs dont have out_edges
                .flat_map(|input| {
                    input
                        .out_edges
                        .borrow()
                        .to_owned()
                        .into_iter()
                        .zip(signals.clone().into_iter())
                })
                .collect();
            //TODO: should never enter the loop
            for i in buffer.iter() {
                // print out_node
                println!(
                    "INIT CYCLE: connection has out_node: {:p}",
                    i.0.get().out_node
                );
            }
            for i in self.outputs.iter() {
                // print out_node
                println!("INIT CYCLE: output node is: {:p}", *i);
            }
            // END OF TEST

            // can still short circuit here if careful due to toplogy constraints
            // (no subset of self.outputs.in_edges will ever equate the set of buffer)
            // TODO: dropped here.
            let mut timeout = 0; // TODO: TEST//
            while !buffer.iter().any(|connection| {
                self.outputs.iter().any(|output| {
                    output
                        .in_edges
                        .borrow()
                        .iter()
                        .any(|final_edge| std::ptr::eq(*final_edge, connection.0))
                })
            }) {
                // TODO: TEST //
                if timeout > 10 {
                    break;
                }
                timeout += 1;
                println!("in the loop with {}", buffer.len());
                // END OF TEST //

                // TODO: dropping buffer
                //iterate a layer of propagation, retaining nodes that dont have all
                //signals yet.
                // NOTE: this is useful for
                //      1. matrix multiplication tensor generation.
                //      2. recurrent connections only require a small change to node
                //         retention condition
                //         and a carry-over signal parameter.
                buffer = buffer
                    .into_iter()
                    // TODO: lolwut
                    .inspect(|x| println!("proc val buff"))
                    .sorted_by_key(|assoc_connection| {
                        assoc_connection.0.get().out_node as *const Node
                    })
                    .group_by(|assoc_connection| assoc_connection.0.get().out_node as *const Node)
                    .into_iter()
                    // TODO: would prefer not to flatten here to reduce sorting per layer
                    //       for retained nodes.
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
                            network::activate_nodes(node_assoc_connections)
                        } else {
                            println!("NODE DELAY");
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
