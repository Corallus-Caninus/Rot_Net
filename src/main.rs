//NOTE: this is primarily for unittests
//TODO: unittest activations::cond_rot_act
//TODO: unittest activations::cond_rot_grad
//TODO: unittest connections::weight
#[macro_use]
extern crate timeit;
use rand::*;

#[cfg(test)]
mod tests {
    use psyclones::psyclones::activations::*;
    use psyclones::psyclones::weights::*;
    //TODO: assert somehow? assert continuous?
    #[test]
    pub fn test_weights() {
        const slopes: [u8; 4] = [
            0b00001000 as u8,
            0b00010000 as u8,
            0b00100000 as u8,
            0b01000000 as u8,
        ];
        println!("TESTING LINEAR BYTE APPROXIMATIONS:");

        println!("~SIGMOID ACTIVATION FUNCTION~");
        //test_cond_rot_act
        for i in 0..255 {
            println!("x = {}, y = {}", i as u8, cond_rot_act(i));
        }

        //test_linear_exponential
        println!("~LINEAR EXPONENTIAL~");
        let mut count = 0;
        for slope in &slopes {
            println!("IN LINEAR_EXPONENTIAL ON SLOPE: {}", count);
            for i in 0..255 {
                println!(
                    "x = {}, y = {}",
                    i as u8,
                    linear_exponential(*slope, i)
                );
            }
            count += 1;
        }
        //test_linear_decay_logarithm
        println!("~LINEAR DECAY LOGARITHM~");
        let mut count = 0;
        for slope in &slopes {
            println!("IN LINEAR_DECAY_LOGARITHM ON SLOPE: {}", count);
            for i in 0..255 {
                println!(
                    "x = {}, y = {}",
                    i as u8,
                    linear_decay_logarithm(*slope, i)
                );
            }
            count += 1;
        }
        //test_linear_logarithm
        println!("~LINEAR LOGARITHM~");
        let mut count = 0;
        for slope in &slopes {
            println!("IN LINEAR_LOGARITHM ON SLOPE: {}", count);
            for i in 0..255 {
                println!(
                    "x = {}, y = {}",
                    i as u8,
                    linear_logarithm(*slope, i)
                );
            }
            count += 1;
        }
        //test_linear_decay_exponential
        println!("~LINEAR DECAY EXPONENTIAL~");
        let mut count = 0;
        for slope in &slopes {
            println!(
                "IN LINEAR_DECAY_EXPONENTIAL ON SLOPE: {}",
                count
            );
            for i in 0..255 {
                println!(
                    "x = {}, y = {}",
                    i as u8,
                    linear_decay_exponential(*slope, i)
                );
            }
            count += 1;
        }
    }
}
//TODO:
//#[test]
//pub fn linear_decay_logarithm() {
//    for i in 0..255 {
//        println!("x = {}, y = {}", i as u8,
// linear_decay_logarithm(i));    }
//}
//#[test]
//pub fn test_cond_rot_act() {
//    for i in 0..255 {
//        println!("x = {}, y = {}", i as u8, (i));
//    }
//}
//#[test]
//pub fn test_cond_rot_act() {
//    for i in 0..255 {
//        println!("x = {}, y = {}", i as u8, cond_rot_act(i));
//    }
//}
fn main() {
    use psyclones::psyclones::rot_net;
    let mut rng = rand::thread_rng();

    println!("Hello, world!");
    let mut rot_net = rot_net::initialize_network(3, 4);
    rot_net.add_node(4, 0, rot_net.tensor.len());
    println!("with new node: {}", rot_net);
    rot_net.add_node(5, 0, rot_net.tensor.len());
    println!("with new node: {}", rot_net);
    rot_net.add_node(7, 0, rot_net.tensor.len());
    println!("with new node: {}", rot_net);

    rot_net.add_connection(5, 7);
    println!("with new node: {}", rot_net);
    rot_net.add_connection(7, 5);
    println!("with new node: {}", rot_net);

    rot_net.add_connection(9, 7);
    println!("with new node: {}", rot_net);
    //TODO: support recurrent connections
    //rot_net.add_connection(4, 4);
    //println!("with new connection {}", rot_net);

    let signals =
        vec![rng.gen::<u8>(), rng.gen::<u8>(), rng.gen::<u8>()];
    let output_signals = rot_net.forward_propagate(signals.clone());

    //timeit!({
    //    rot_net.forward_propagate(signals.clone());
    //});
    //NOTE: initial nodeIds are out of order
    println!(
        "forward propagating {:?} returned {:?}",
        signals.clone(),
        output_signals
    );
    println!("{}", rot_net);
    let time = timeit_loops!(10000, {
        rot_net.forward_propagate(signals.clone());
    });
    let size = std::mem::size_of_val(&rot_net);
    println!("rot_net clocked @ {} \n with {} bytes", time, size);
}
