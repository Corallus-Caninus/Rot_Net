use rotnet::*;
// use net::connection::*;
use rand::{Rng, SeedableRng};
use rotnet::rot_net::*;
use rotnet::rot_net::*;
use rotnet::*;
use std::cell::{Cell, RefCell};
use std::{boxed::Box, ops::DerefMut};
#[macro_use]
extern crate timeit;

#[cfg(test)]
mod tests {
    // use net::*;
    use rotnet::rot_net::*;
    use std::cell::{Cell, RefCell};
    // TODO: profile before using a different non crypto PRNG.
    //       True entropy is more important than is intuitive for search.
    use rand::{Rng, SeedableRng};

    #[test]
    pub fn construct_rot_net() {
        let mut rng = rand::thread_rng();
        let num_inputs = 3;
        let num_outputs = 2;

        // 1. initialize nodes
        let nodes = Network::initialize_nodes(num_inputs, num_outputs);
        // TODO: &Nodes needs to fall off here!
        let extremas = Network::initialize_extrema(&nodes, num_inputs, num_outputs);
        let inputs = extremas.0;
        let outputs = extremas.1;

        // 2. initialize connections
        let connections = Network::initialize_connections(&inputs, &outputs);
        for connection in connections.iter() {
            println!(
                "verify connection out_node edges: {:p}",
                connection.get().out_node
            );
        }

        // 3. initialize rot_net
        //Network::initialize_Network(&inputs, &outputs, &connections.iter().collect());
        // TODO: just need to drop the borrow on initialize_extrema
        // TODO: just let inputs and outputs own extrema nodes? this feels like quitting..
        let mut rot_net = Network {
            inputs: inputs,
            outputs: outputs,
            hidden_nodes: vec![],
        };
        println!("initializing out_connections in Network..");
        rot_net.initialize_network_out_connections(connections);

        println!("CONSTRUCTED TOPOLOGY METRICS");
        for i in rot_net.inputs.iter() {
            println!("this input has {}", i.out_edges.borrow().len());
        }
        for o in rot_net.outputs.iter() {
            println!("got topology out_node: {:p}", *o);
        }
        // TODO: copy this template as another unittest and call basic
        //       complexifying routines.
        // 4. cycle the Network
        println!("cycling Network..");
        let res = rot_net.cycle(vec![rng.gen::<u8>(), rng.gen::<u8>(), rng.gen::<u8>()]);
        for r in res {
            println!("got result: {}", r);
        }
    }
}

fn main() {
    println!("Hello, world!");
}
