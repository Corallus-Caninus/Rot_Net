//NOTE: this is primarily for unittests
//TODO: unittest activations::cond_rot_act
//TODO: unittest activations::cond_rot_grad
//TODO: unittest connections::weight

#[cfg(test)]
mod tests {
    use psyclones::psyclones::activations::*;
    use psyclones::psyclones::connections::*;
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
                println!("x = {}, y = {}", i as u8, linear_exponential(*slope, i));
            }
            count += 1;
        }
        //test_linear_decay_logarithm
        println!("~LINEAR DECAY LOGARITHM~");
        let mut count = 0;
        for slope in &slopes {
            println!("IN LINEAR_DECAY_LOGARITHM ON SLOPE: {}", count);
            for i in 0..255 {
                println!("x = {}, y = {}", i as u8, linear_decay_logarithm(*slope, i));
            }
            count += 1;
        }
        //test_linear_logarithm
        println!("~LINEAR LOGARITHM~");
        let mut count = 0;
        for slope in &slopes {
            println!("IN LINEAR_LOGARITHM ON SLOPE: {}", count);
            for i in 0..255 {
                println!("x = {}, y = {}", i as u8, linear_logarithm(*slope, i));
            }
            count += 1;
        }
        //test_linear_decay_exponential
        println!("~LINEAR DECAY EXPONENTIAL~");
        let mut count = 0;
        for slope in &slopes {
            println!("IN LINEAR_DECAY_EXPONENTIAL ON SLOPE: {}", count);
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
//        println!("x = {}, y = {}", i as u8, linear_decay_logarithm(i));
//    }
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
    use psyclones::psyclones::net_index;
    println!("Hello, world!");
    let mut rot_net = rot_net::initialize_rot_net(3, 2);//TODO: this isnt correct.
    let in_node = net_index{layer:0, node:0};
    let out_node = net_index{layer:0, node:3};
    rot_net.add_split_node(in_node, out_node);

    println!("rotation network construction complete {}", rot_net);
}
