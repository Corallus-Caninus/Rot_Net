#[macro_use]
extern crate timeit;
use std::env;

// TODO: run a Jenkins build pipeline or something for these unittests for DEVOPs.
//       scale this out into application (gradient architecture search and BWAPI)
#[cfg(test)]
mod tests {
    use log::info;
    use net::activations::*;
    use net::connections::*;

    #[test]
    fn test_activation() {
        print!("Testing of sigmoid activation of 8 bit integer using only bitshift operations.");
        print!("target runtime is 3 clock cycles for 3 barrel shift operations.\n");
        for i in 1..255 {
            print!(
                "\n sent {} and got: {} == {}",
                i,
                format!("{:b}", cond_rot_act(i)),
                cond_rot_act(i)
            );
        }
        //TODO: check distance between i and i+1 to ensure no piecewise discontinuity
        print!("Testing conditional rotation activation..");
        timeit!({
            for i in 1..255 {
                cond_rot_act(i);
            }
        });
        print!("Testing unoptimized full 32x precision sigmoid activation..");
        timeit!({
            for i in 1..255 {
                (1.0_f32 / 1.0_f32 + (-1.0_f32 * (i as f32)).exp());
            }
        });
        print!("Testing multiplication..");
        timeit!({
            for i in 1..255 {
                i * (i + 1);
            }
        });
    }
    // #[test]
    // fn test_connection_propagation() {
    //     print!("Testing forward propagation of a connection with 8 bit shift operations");
    //     let connection = Connection {
    //         weight: 4 as u8,
    //         in_connection: None,
    //         out_connection: None,
    //     };
    //     for i in 1..255 {
    //         let solution = connection.forward_propagate(i as u32);
    //         // res = connection.propagate()
    //         print!("\n sent {} and got {}", i, solution);
    //     }
    // }
    #[test]
    fn test_network_forward_propagation() {
        print!("Testing forward propagation of a network");
        // TODO: create network and forward propagate
    }
}

//CURRENTLY TESTING WITH PRINTOUT:
// activation.
fn main() {
    use net::connections::*;
    use net::network::Network;
    env::args().for_each(|x| println!("got {}", x));
    let mut args: Vec<String> = env::args().collect();
    args.remove(0);
    let args = args
        .iter()
        .map(|x| x.parse::<u32>().unwrap())
        .collect();

    let output_connection = &Connection {
        weight: 4 as u8,
        in_connection: None,
        out_connection: None,
    };
    let input_connections = vec![
        Connection {
            weight: 8,
            in_connection: None,
            out_connection: Some(output_connection),
        },
        Connection {
            weight: 4,
            in_connection: None,
            out_connection: Some(output_connection),
        },
    ];
    // now move connections with ownership into network for RAII acquisition
    let mut test_net = Network {
        connections: vec![output_connection],
        inputs: input_connections,
    };

    test_net
        .inputs
        .iter()
        .for_each(|x| println!("INPUT ADDRESS: {:p}", x));
    println!("OUTPUT ADDRESS: {:p}", output_connection);
    let answer = test_net.forward_propagate(args);
    for a in answer {
        println!("got {}", a);
    }
}
