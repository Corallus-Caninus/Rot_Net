use std::vec;

//NOTE: These are the droids you're looking for.
//32 bit connections
pub mod connections {
    // going with nodeless topology, activations must occur functionally
    // TODO: network object holds connection boxes? Can ref count here but..
    pub struct Connection<'a> {
        weight: u8,
        direction: bool,
        in_connection: &'a Connection<'a>,
        out_connection: &'a Connection<'a>,
    }
    // TODO: attempt normalized connections. after sum normalize prior to activation
    //       (squash then activate-- activation is for network emergent approximation not squashing)
    //       this would mean the entire network is u8 and dont need 32 buffer extension which doesnt
    //       carry representation anyways without a larger activation domain.
    impl<'a> Connection<'a> {
        pub fn propagate(&mut self, signal: u8) -> u8 {
            //implement with rotations and/or masking.
            //This is the primary network bottleneck not activation (secondary).
            if self.direction {
                signal << self.weight
            } else {
                signal >> self.weight
            }
        }
    }
}
pub mod activations {
    //      uses 32 bit half prec.
    //      hard code a limit to max connections in architecture search
    //      because vectorization is more important at that point than 128 etc. precision.

    //  besides NEAT/AS, res and recurrent net that are fast may be
    //  really useful in general

    //      consider normalized connections so all values lie within sigmoid feature curves.
    //      this may take away from speed (somewhat.. maybe) but makes the learning much faster

    //NOTE: based on my understanding of DL and information theory, this is
    //      feature complete with sigmoid. There is no reason to use sigmoid function
    //      and the binning of values outside of sigmoid "sweet spot" innate to float
    //      mantissa operation seems like a bar trick more than a feature. This is why
    //      RELU took over all but the output.
    //  binning the activation function instead and using binary operations makes more sense
    //  from a computer engineering standpoint. Can be discretized and old hardware makes
    //  newer stuff useless. hash the input and send it like a 8055. all inputs are finite
    //  and usually discretized (in time, space or sequence) anyways.

    //TODO: solve gradient USING THE APPROXIMATION.
    //      you cannot approximate the gradient of the original function! compounding error!
    //      dr.dzyubenko would be dissapointed...
    // gradient of linear approximation is piecewise continuous constants! Its free!
    // get creative people..
    //TODO: still need connection rot gradient. This may be tricky since bitshift is lossy.

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

    // TODO: would this precision be as fast with a LUT given binning?
    //       this is to be scaled out in the domain to >255 anyways
    //      for more representation so probably not.
    // TODO: expand domain to some value in u32 > u8. calculate connections required to sum to this.
    //       possibly u32? u16 is a good medium u32 will never be reached, requires tens of thousands
    //       of fully activated connections

    /// a conditional piecewise sigmoid approximating sigmoid with 5 line segments
    /// and using bitshifts for slopes (it's fast-- real fast) ...(can it be faster.?)
    pub fn cond_rot_act(x: u32) -> u8 {
        const SEGMENTS: [u32; 5] = [30, 75, 128, 170, 255];
        //SEGMENTATION DESCRIPTION:
        // 1. tail of sigmoid is somewhat arbitrarily set @ 30 which is 10% of domain.
        // 2. head of sigmoid is all values >255
        // The lines and intercepts are solved for two slopes: 1/2 and 2 given 1. and 2.
        // This approximation is the maximum precision and minimum error optimizing for speed

        //TODO: analyse whether these are efficient on broadwell given having to offset by y-intercept
        //      if not, check if sandybridge/bulldozer would benefit given its reduced vectorization
        //      (add-sum isnt vectorized? was introduced in haswell)

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

    //TODO: These arent functional. Thar be dragons.
    //  left for documenting the research process.
    /// very simple linear sigmoidal function with mirrored and shifted relus.
    /// Is asymmetric but that is more expressive (bi-modal).
    ///
    /// squashes 32 to 8 bit precision.
    ///
    /// sadly this is the best yet. conditional prevents base2 oscillations.
    ///
    /// NOTE: ANN implementations will likely get some expression out of this but chopping off bits as values
    ///       increase is identical to mantissa loss of precision in fp32 and is a design avoidance
    ///       in further development.
    //TODO: this can be elongated by considering every other bit and binning to prevent oscillation
    //      (constant for every other bit regardless of lower bits in word)
    //      this is still linear but require just a few additional bitshifts to give curvature.
    //      curvature will be discrete (binned) but still folow power rule and give good
    //      representation considering 8 bit limitations.
    fn rot_act(x: u32) -> u32 {
        //TODO: start with this and develop exponential and logarithmic curves.
        //      there are some bool algebra tricks that make this possible I
        //      just need to figure them out.
        //let mask = 0x0ffffff0;
        const MSB_mask: u32 = 0xf0000000;
        //TODO: can scoot these over to get less sensitive expression. requires
        //      binning to prevent base2 oscillation.
        const LSB_mask: u32 = 0x0000000f;
        const mask: u32 = 0x0ffffff0;

        let lossy_squash = x & MSB_mask;
        let LSB = x & LSB_mask;
        let line = x & mask;
        //return constant value if between 2^4 and 2^7
        //TODO: check conditions based on segment size assuming a gaussian probability v 1
        //      this will minimize worst case comparisons.
        if lossy_squash == 0x0 && line != 0x0 {
            127
        //return low slope line if >=2^7
        } else if lossy_squash != 0x00 {
            //TODO: offset by 2^7 to prevent redundant expressivity.
            let lossy_squash = lossy_squash >> 24;
            //offset by 2^7 to prevent piecwise disconnect
            let res = lossy_squash | 0x80;

            res
        }
        //return linear activation if under 2^4
        else {
            LSB
        }
    }

    /// squashes 32 bits. doesnt allow for more than a couple hundred connections
    /// due to worse case overflow and requires 32 bit parameters and buffer.
    ///
    /// TODO: may require considerable backprop compute. check boolean
    ///       algebra differentiators.
    /// NOTE: I need to practice hexadecimal arithmetic. just think in terms of 4 bytes
    fn old_rot_act(x: u32) -> u32 {
        //TODO: shift this down s.t. rot_act(x)_max <= 255 = 2^8
        // 32 bit squishing of MSBs (u32 precision)
        // strong powers decay of MSBs
        let res = x & 0xff000000; // grab the upper MSB that will set the squash
        println!("32BIT INDEX ROT_ACT: {}", format!("{:b}", res));
        let res = res >> 27; //shift to 0-31 bit rotation.
        println!("32BIT INDEX ROT_ACT: {}", format!("{:b}", res));
        let res = x >> res; // bitshift solution (powers of 2 decay)
        println!(
            "32BIT INDEX FINAL ROT_ACT: {} => {}",
            format!("{:b}", res),
            res
        );
        //let res = x >> 1; //simple but doesnt squash enough. trace this.
        // TODO: this is simple just do the 32 bit shift again for 16
        // 8 bit squishing (u16 precision)
        // TODO: once this is figured out. diagnose a sigmoid from 0 to max(uint8)
        //       this is a weird repeating line with decay. 3 modes is good but not
        //       if they map to the exact same output.
        //       Dont make this as complicated. each inflection point changes shift direction.
        //       otherwise sigmoid is a powers curve.
        // has a strange perturbation for every other bit power.
        // TODO: dont shift 32 bits just 16
        // TODO: this needs to be a loop and shift every other bit in 2s or something.
        //bad barrel technique as outlined
        res
    }
    //TODO: rot connection activation using XOR and possibly 2x bitshift (1 vs 2 clocks)
}
use crate::connections::Connection;

pub mod network {
    pub struct Network<'a> {
        // TODO: consider something more optimal than vec such as numpy array
        connections: Vec<crate::connections::Connection<'a>>,
        //inputs and outputs hold Refs therefor need RefCell
        inputs: Vec<crate::connections::Connection<'a>>,
        outputs: Vec<crate::connections::Connection<'a>>,
    }
    impl<'a> Network<'a> {
        //forward propagate through the network
        pub fn cycle(input: Vec<u8>, output: Vec<u8>) {
            //TODO: take a u8 array and return a u8 array
            let buffer = vec![input.len()];
            for i in input.iter(){
            }
        }
        pub fn add_connection(&mut self, addition: crate::connections::Connection) {
            //TODO: add a connection to self.
            //NOTE: no add_node or mutation methods since this is more high level
            //      k.stanley suggests in emails topology is represented as connections
            //      and this is more space efficient.
            //  consequently this can create hanging nodes and parallel connections.
        }
    }
}
