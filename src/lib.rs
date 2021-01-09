//NOTE: These are the droids you're looking for.
//32 bit connections
pub mod connections {
    // going with nodeless topology, activations must occur functionally
    // TODO: network object holds connection boxes? Can ref count here but..
    pub struct Connection<'a> {
        pub weight: u8,
        pub direction: bool,
        // TODO: optionals here for frontier nodes.
        //       these are called in network methods so pub here.
        pub in_connection: Option<&'a Connection<'a>>,
        pub out_connection: Option<&'a Connection<'a>>,
    }
    // TODO: attempt normalized connections. after sum normalize prior to activation
    //       (squash then activate-- activation is for network emergent approximation not squashing)
    //       this would mean the entire network is u8 and dont need 32 buffer extension which doesnt
    //       carry representation anyways without a larger activation domain.
    // TODO: this repeats due to bit shift since 0 overflow shift implemented.
    //       set max weight value to index precision or change otherwise.
    //       (scrubbing (notch_rot?) has combination of data type length distinct
    //       outputs f(x) != {f({y})} | x !e y: len(dtype)!)
    //       can conditional inequality piecewise scrubbing. This yields a smooth function
    //       that can be conditionally shaped to saddle points. at what condition+binop count
    //       is mult faster (at scale, considering SMD etc)? is this faster than unsigned integer
    //       multiplication because thats what is is unless creating a function that isnt a saddle.
    // TODO: 16 conditions for non isotropic function(continually increasing/decreasing).
    //       mirror 8 conditions for a saddle or smoother quadratic curve (convex)
    // TODO: 0 shifting is unreversible (lossy). cant readily take the gradient. can do the gradient
    //       piecewise trick like cond_rot_sigmoid
    // NOTE: for now, multiplication is plenty fast and well optimized.
    impl<'a> Connection<'a> {
        pub fn forward_propagate(&self, signal: u32) -> u32 {
            self.weight as u32 * signal
            // TODO: pub fn backward_propagate() { } //for multiplication this is typical subtraction
        }
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

    //TODO: solve gradient USING THE APPROXIMATION.

    ///calculate the activation gradient from an error metric
    pub fn cond_rot_grad(x: u32) -> u8 {
        //TODO: global extract static
        const SEGMENTS: [u32; 5] = [30, 75, 128, 170, 255];
        // TODO: slopes must be doubled since this creates a deriv of 1/2?
        //      this should be correct. just check back when implementing.
        //      can just use a slope of 1 and 2 why 1/2 anyways? 1 and 4 for
        //      better approximation of a given sigmoid
        if x < SEGMENTS[0] {
            0
        } else if x < SEGMENTS[1] {
            1
        } else if x < SEGMENTS[2] {
            2
        //GAUSSIAN POINT OF CONCAVITY//
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
        //    not buffer precision, this may change.
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
    use crate::connections::*;
    pub struct Network<'a> {
        // TODO: consider something more optimal than vec such as numpy array
        connections: Vec<Connection<'a>>,
        //inputs and outputs hold Refs therefor need RefCell
        inputs: Vec<Connection<'a>>,
        outputs: Vec<Connection<'a>>,
    }
    ///a very simple artificial neural network class.
    impl<'a> Network<'a> {
        /// insert a connection into this network. This is the fundamental operation and isnt
        /// checked for nodes. This can create subtrees that dont trace to inputs or outputs
        /// so it is expected to be wrapped and called appropriately from caller.
        pub fn add_connection(&mut self, _addition: crate::connections::Connection) {
            //TODO: add a connection to self.
            //NOTE: no add_node or mutation methods since this is more high level
            //      k.stanley suggests in emails topology is represented as connections
            //      and this is more space efficient.
            //  consequently this can create hanging nodes and parallel connections.
        }
        /// takes the given inputs and propagates the signals through the network,
        /// returning the final signals at the output. every node is activated once,
        /// creating one-pass or "per pass" recurrence. Although simple, this can
        /// describe any unrolled neural network and also allows for res-net features.
        pub fn cycle(&mut self, inputs: Vec<u8>) -> Vec<u8> {
            //TODO: async threadpool layers since will need to scale very large.
            let output: Vec<u8> = vec![];
            let buffer: Vec<u32> = vec![];
            //TODO: propagate and return
            output
        }
    }
}
