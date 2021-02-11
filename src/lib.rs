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
    use activations::cond_rot_act;
    use itertools::Itertools;
    use rand::Rng;
    use std::{
        borrow::BorrowMut,
        cell::{Cell, RefCell},
    };

    //TODO: use generics to ingest any data type and automatically vectorize larger precision types/classes into self.inputs
    //      cast to bits and discretize/bin? want to be able to send any object and chunk/vectorize into a network.
    //      std::mem::transmute might be a good option here. some methods within serde might also be helpful.
    //     use same generic casting schema for the output harvesting.
    //TODO: can we center all data points on ingest to 147 and implement a psuedo 0 crossing sigmoid
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
        //TODO: consolidate all these methods for construction to one that is called in owning scope
        /// create a fully connected topology with input and output
        /// nodes and randomized connections.
        pub fn rot_net_initialize(inputs: Vec<&'a Node<'a>>, outputs: Vec<&'a Node<'a>>) -> Self {
            network {
                inputs: inputs,
                outputs: outputs,
            }
        }
        /// returns the initial connections given the output nodes.
        /// THIS IS TODO: remove
        pub fn connect_initial_network(
            outs: Vec<&'a Node<'a>>,
            rng: Vec<u8>,
        ) -> Vec<Cell<Connection>> {
            let mut res = vec![];
            for i in outs.iter().zip(rng) {
                let cur = Connection {
                    out_node: i.0,
                    params: i.1,
                };
                res.push(Cell::new(cur));
            }
            res
        }
        // TODO: add connections in another function call..
        pub fn initialize_network(
            num_inputs: i64,
            num_outputs: i64,
            connections: Vec<&'a Cell<Connection<'a>>>,
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
            inputs.iter().zip(connections.iter()).for_each(|ins| {
                ins.0.out_edges.borrow_mut().push(ins.1);
            });
            outputs
                .iter()
                .zip(connections.into_iter())
                .for_each(|outs| outs.0.in_edges.borrow_mut().push(outs.1));
            (inputs, outputs)
        }
        pub fn add_random_connection(self) {}
        pub fn add_random_node(self) {}
        // IMMUTABLE OPERATIONS //
        // TODO: at some point this will be matrix ops instead
        // TODO: ENSURE CLONE USES COPY FOR OWNED CONNECTIONS
        /// forward propagate.
        /// consumes signals and returns the output vector.
        pub fn cycle(self, mut signals: Vec<u8>) -> Vec<u8> {
            //buffer contains the current node layer in_connections.
            //each iteration sum squashes and broadcasts returning
            //the resultant broadcast connection-signal tuples

            // TODO: verify not comparing Cell<Connection> but Connection
            // Initialize: get signals into the topology and ready the first layer
            //  1. each input node gets a signal and broadcasts
            let mut buffer: Vec<(u8, Connection)> = self
                // initialize
                .inputs
                .iter()
                // TODO: is this to_owned pointer safe?
                .map(|x| x.out_edges.borrow().to_owned())
                .flatten()
                // initial input weighting routine: //
                .group_by(|x| x.get().out_node as *const _)
                .into_iter()
                .map(|(_key, group)| {
                    group
                        .into_iter()
                        .map(|x| {
                            // hillbilly len assertion:
                            let cur_signal = signals.pop().unwrap();
                            (x.get().weight(cur_signal), x.get().out_node)
                        })
                        .collect::<Vec<(u8, &Node)>>()
                })
                .flatten()
                // sum-squash-activation routine: //
                .group_by(|x| x.1 as *const _)
                .into_iter()
                .map(|(_key, group)| {
                    // TODO: can this be done without mutables?
                    // TODO: make this lazy without for_each? Cant this be threaded?
                    let mut proc_sig = 0 as u16; // twice precision for best fit of worst case 255+255
                    let mut node_buffer: Option<&Node> = None;
                    group.into_iter().for_each(|x| {
                        // this is now normalized.
                        proc_sig += x.0 as u16;
                        proc_sig = proc_sig >> 1;
                        // TODO: FIX THIS x.1 is the same throughout because it got grouped.
                        //       how is there not a better way to key by object reference??
                        node_buffer = Some(x.1);
                    });
                    (
                        activations::cond_rot_act(proc_sig as u8),
                        node_buffer.unwrap(),
                    )
                })
                .map(|x| {
                    // TODO: this dereference so something should be into_iter close above
                    // Broadcast routine: //
                    x.1.out_edges
                        .borrow()
                        .iter()
                        .map(|e| (x.0, e.get()))
                        .collect::<Vec<(u8, Connection)>>()
                })
                .flatten()
                // TODO: this should all be able to be lazy
                .collect();

            println!("INITIALIZATION COMPLETE");

            // Propagate: forward propagate hidden layers //
            // TODO: this short circuits but is still a little wonky
            while buffer
                .iter()
                .map(|x| x.1.out_node)
                .all(|x| self.outputs.iter().any(|y| std::ptr::eq(*y, x)))
            {
                println!("IN THE LOOP");
                // TODO: REFACTOR. this is mainly copy pasta from initialization routine
                buffer = buffer
                    .iter()
                    // weight the connections using buffer signals
                    .map(|x| (x.1.weight(x.0), x.1))
                    .group_by(|x| x.1.out_node as *const _)
                    .into_iter()
                    // TODO: refactor this functionally as node activation
                    // TODO: delay node condition
                    .map(|(_key, group)| {
                        // TODO: can this be done without mutables like this?
                        // TODO: make this lazy without for_each? Cant this be threaded?
                        // dont like collecting like this but until Node can implement PartialEq or can group by object reference..
                        let g = group.collect::<Vec<(u8, Connection)>>();
                        // reference for comparator and pass by value for carry operation
                        if g.iter().all(|x| {
                            x.1.out_node
                                .in_edges
                                .borrow()
                                .iter()
                                .any(|y| std::ptr::eq(&x.1, &y.get()))
                        }) {
                            // Node Activation and Broadcast routines: //
                            let mut proc_sig = 0 as u16; // twice precision for best fit of worst case 255+255
                            let mut node_buffer: Option<&Node> = None;
                            g.into_iter().for_each(|x| {
                                // this is now normalized.
                                proc_sig += x.0 as u16;
                                proc_sig = proc_sig >> 1;
                                // TODO: FIX THIS x.1 is the same throughout because it got grouped.
                                //       how is there not a better way to key by object reference??
                                node_buffer = Some(x.1.out_node);
                            });
                            let proc_sig = activations::cond_rot_act(proc_sig as u8);
                            // TODO: get to keys this only works as unsafe raw deref
                            // TODO: broadcast here instead
                            node_buffer
                                .unwrap()
                                .out_edges
                                .borrow()
                                .iter()
                                .map(|x| (proc_sig, x.get()))
                                .collect::<Vec<(u8, Connection)>>()
                        } else {
                            // Node delay routine: //
                            g.iter()
                                .map(|x| (x.0, x.1))
                                .collect::<Vec<(u8, Connection)>>()
                        }
                    })
                    .flatten()
                    .collect::<Vec<(u8, Connection)>>();
            }
            // TODO: collect output signals.
            buffer
                .iter()
                .group_by(|x| x.1.out_node as *const _)
                .into_iter()
                .map(|(_key, group)| {
                    // all signals must be ready so sum squash unconditionally.
                    group
                        .into_iter()
                        .map(|x| x.0)
                        .fold(0, |sum, x| sum + x as u16 >> 1)
                })
                .map(|x| activations::cond_rot_act(x as u8))
                .collect::<Vec<u8>>()

            // //TODO: REFACTOR EXTRACT THESE ROUTINES
            // 1. weighted,
            // 2. grouped by node,
            // 3. sum-normalized,
            // 4. activated,
            // 5. broadcast to next connections and returned as a (signal:u8,Connection)
        }
    }
}
