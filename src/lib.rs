//NOTE: These are the droids you're looking for.

//32 bit connections
pub mod connections {

    // #[derive(PartialEq)] //TODO: consider deriving for publication
    pub struct Connection<'a> {
        //24 bytes per parameter without explicit nodes
        //network also holds an additional reference so + addressable_size*parameters
        pub weight: u8, //8 bytes
        // Rust has great option space optimization so we will use it when possible:
        //  sizeof(Option<&U>) == sizeof(&U)
        pub in_connection: Option<&'a Connection<'a>>, //8 bytes
        pub out_connection: Option<&'a Connection<'a>>, //8 bytes
    }
}
pub mod activations {
    //      uses 32 bit prec.
    //      hard code a limit to max connections in architecture search or implementation.
    //      TODO: consider half prec because vectorization is more important at that point
    //            than higher precision.

    //      consider normalized connections so all values lie within sigmoid feature curves.
    //      this may take away from speed (somewhat.. maybe) but makes the learning much faster.
    //      see: quantization/normalization in ANNs for more (well published)
    ///calculate the activation gradient from an error metric
    pub fn cond_rot_grad(x: u32) -> u8 {
        //TODO: global extract static
        const SEGMENTS: [u32; 5] = [30, 75, 128, 170, 255];
        // TODO: slopes must be doubled since this creates a deriv of 1/2?
        //      this should be correct. just check back when implementing.
        //      can just use a slope of 1 and 2 why 1/2 anyways? 1 and 4 for
        //      better approximation of a given sigmoid.

        // TODO: precision slope these in a given direction e.g. into point of inflection {>,>,<,<}
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

    // TODO: expand domain to some value in u32 > u8. calculate connections required to sum to this.
    //       possibly u32? u16 is a good medium u32 will never be reached, requires tens of thousands
    //       of fully activated connections

    /// approximates a sigmoid with 5 line segments for inflection points and
    /// bitshifts for slopes (it's fast-- real fast) ...(can it be faster.?)
    pub fn cond_rot_act(x: u32) -> u8 {
        const SEGMENTS: [u32; 5] = [30, 75, 128, 170, 255];
        //SEGMENTATION DESCRIPTION:
        // 1. tail of sigmoid is somewhat arbitrarily set @ 30 which is 10% of domain.
        // 2. head of sigmoid is all values >255, NOTE: sigmoid spans parameter precision
        //    not cur_buffer precision, this may change.
        // The lines and intercepts are solved for two slopes: 1/2 and 2 given 1. and 2.
        // This approximation is the maximum precision and minimum error optimizing for speed

        //TODO: analyse whether these are efficient on broadwell given having to offset by y-intercept
        //      if not, check if sandybridge/bulldozer would benefit given its reduced vectorization
        //      (add-sum isnt vectorized? was introduced in haswell. bulldozer has AVX but not AVX2 (double wide?))

        // TODO: should be a case but w/e
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

pub mod network {
    extern crate itertools;
    use crate::activations::*;
    use crate::connections::*;
    use itertools::Itertools;

    pub struct Network<'a> {
        pub connections: Vec<&'a Connection<'a>>,
        //extrema nodes
        //NOTE: inputs are connections without in_connections
        pub inputs: Vec<Connection<'a>>,
        // NOTE: outputs are connections without out_connections
        //@DEPRECATED
        // outputs: Vec<Connection<'a>>,
    }

    /// a very simple artificial neural network class that attempts
    /// to be space efficient using lower precision and conditional
    /// approximation with binops. Allows residual connections or
    /// "skip" connections and may implement recurrent connections
    /// in the future.
    impl<'a> Network<'a> {
        /// insert a connection into this network.
        /// This can create subtrees that dont trace to inputs or outputs
        /// so it is expected to be wrapped and called appropriately from caller.
        pub fn add_connection(&mut self, addition: &'a Connection<'a>) {
            //TODO: assert connections are referenced properly
            self.connections.push(addition);
        }

        /// Takes the given inputs and propagates signals through the network,
        /// returning the final signals at the outputs. currently with res net architecture.
        ///
        /// recurrent connections are TODO (maybe.. equivalent representation (e.g.: dot product transformer vs resnet)
        /// is a research topic)
        ///
        /// inputs must be positionally encoded (matched) to nodes.
        /// changing positions in the vector will skew input and destabilize trainning.
        pub fn forward_propagate(&mut self, inputs: &mut Vec<u32>) -> Vec<u8> {
            //TODO: assert_eq!(inputs.len(), self.inputs.len())
            //TODO: input can be stack array [u32; self.input.len()] ?
            //TODO: shouldnt need to borrow mutably here.

            //NOTE: prefer using iterator method chaining (map-reduce) here in preference
            //      to for{} for lazy parallelization with async threadpools and possibly
            //      implementing different iterator methods (its functional).
            //      this looks like alot but by keeping signals in the cur_buffer here we can
            //      avoid the memory overhead of signals in parameter space.

            // let output: Vec<u8> = vec![];
            //TODO: threadpool/rayon. Ensure global threadpool is configured if par_iter chaining.
            //      (node_cur_buffer, signal_buffer)

            //initializing connection and signal associated tuples with input vector
            let mut buffer: Vec<(&Connection, u8)> = self
                .inputs
                .iter()
                .map(|x| {
                    (
                        x, //the node in question
                        //move in and consume input vector with squash normalization,
                        cond_rot_act(x.weight as u32 * inputs.pop().unwrap() as u32), //it's associated signal
                    )
                })
                .collect();
            print!("buffer intialized..");

            //loop until output vector is ready, since this isnt recurrent we are done when the buffer == self.outputs
            while buffer.iter().any(|x| x.0.out_connection.is_some()) {
                // TODO: if possible dont collect so much for lazy optimization
                //COLLECT COUNT: 6
                // TODO: trim references throughout iterators. probably optimized with backend but dont assume.
                // TODO: get precision right. Should be all u32 here unless normalizing. signals are u8
                // TODO: sorting here would yield optimal performance after +1 forward propagation
                //       (prepare network before forward propagating which works with NEAT genotype->phenotype representation)

                println!("\n LAYER: in the loop.."); //TODO: dump buffer
                buffer.iter().for_each(|x| {
                    println!("LAYER: got buffer value: {:p}", x.0);
                    println!(
                        "LAYER: got buffer out_connection value: {:p} and {}",
                        x.0.out_connection.unwrap(),
                        x.0.out_connection.is_none()
                    )
                });

                // TODO: move this out, may need a function call here which would be more organized anyways.
                buffer = buffer
                    .into_iter()
                    .filter(|x| x.0.out_connection.is_some())
                    .collect();

                buffer = buffer
                    .into_iter()
                    // we can unwrap because of previous option filter
                    .group_by(|x| x.0.out_connection.unwrap() as *const _)
                    .into_iter()
                    .map(|(key, group)| {
                        //collect "node" groups here so we can move by reference
                        let group: &Vec<(&Connection, u8)> = &group.collect();

                        // TODO: not checking out_connections in group items
                        if group.iter().all(|g| {
                            self.connections
                                .iter()
                                .filter(|c| std::ptr::eq(**c as *const _, key))
                                .any(|c| std::ptr::eq(*c as *const _, g.0.out_connection.unwrap()))
                        }) {
                            // TODO: this dont happen
                            println!("NODE: activation routine..");
                            group
                                .iter()
                                .map(|c| {
                                    (
                                        c.0.out_connection.unwrap(),
                                        // this is the only place where buffer is u32 appropriately because sum overflow
                                        cond_rot_act(group.into_iter().map(|x| x.1 as u32).sum()),
                                    )
                                })
                                .collect::<Vec<(&Connection, u8)>>()
                        } else {
                            println!("NODE: delay routine..");
                            group
                                .into_iter()
                                .map(|x| (x.0, x.1 as u8))
                                .collect::<Vec<(&Connection, u8)>>()
                        }
                    })
                    .flatten()
                    .collect();
            }
            //TODO: move this correctly
            let output = buffer.iter().map(|connection| connection.1 as u8).collect();
            output
        }
    }
}
