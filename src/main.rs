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
    use std::cell::Cell;
    // TODO: profile before using a different non crypto PRNG.
    //       True entropy is more important than is intuitive for search.
    use rand::{Rng, SeedableRng};
    // #[test]
    // pub fn construct_rot_net() {
    //     let mut rng = rand::thread_rng();
    //     // 1. initialize nodes
    //     let num_inputs = 3;
    //     let num_outputs = 2;
    //     let nodes = network::initialize_nodes(num_inputs, num_outputs);

    //     // 2. initialize rot_net
    //     let extrema_nodes = network::extrema_nodes(nodes.iter().collect(), num_inputs, num_outputs);
    //     let rot_net = network::initialize_rot_net(nodes, extrema_nodes.0, extrema_nodes.1);

    //     // 3. initialize connections
    //     let init_connections = network::initialize_connections(rot_net.inputs, rot_net.outputs);
    //     let connections = init_connections.iter().collect::<Vec<&Cell<Connection>>>();

    //     // 4. cycle the network
    //     let res = rot_net.cycle(vec![1 as u8, 2 as u8, 3 as u8]);
    //     for r in res {
    //         println!("got result: {}", r);
    //     }
    // }
}

fn main() {
    println!("Hello, world!");
    let num_inputs = 3;
    let num_outputs = 2;

    // 1. initialize nodes
    let nodes = network::initialize_nodes(num_inputs, num_outputs);
    // TODO: &Nodes needs to fall off here!
    let extremas = network::initialize_extrema(&nodes, num_inputs, num_outputs);
    let inputs = extremas.0;
    let outputs = extremas.1;

    // 2. initialize connections
    let connections = network::initialize_connections(&inputs, &outputs);
    //TODO: add connections to nodes

    // 3. initialize rot_net
    //network::initialize_network(&inputs, &outputs, &connections.iter().collect());
    // TODO: just need to drop the borrow on initialize_extrema
    // TODO: just let inputs and outputs own extrema nodes? this feels like quitting..
    let rot_net = network {
        inputs: inputs,
        outputs: outputs,
        hidden_nodes: vec![],
        connections: connections,
    };
    rot_net.initialize_network();

    for o in rot_net.outputs.iter() {
        println!("got topology out_node: {:p}", *o);
    }
    for c in rot_net.connections.iter() {
        println!("got connection out_node: {:p}", c.get().out_node);
    }
    // 4. cycle the network
    println!("cycling network..");
    let res = rot_net.cycle(vec![1 as u8, 2 as u8, 3 as u8]);
    for r in res {
        println!("got result: {}", r);
    }
}
