use net::*;
// use net::connection::*;
use net::rot_net::*;
use net::rot_net::*;
use net::*;
use rand::{Rng, SeedableRng};
use std::cell::{Cell, RefCell};
use std::{boxed::Box, ops::DerefMut};
#[macro_use]
extern crate timeit;

#[cfg(test)]
mod tests {
    // use net::*;
    use net::rot_net::*;
    use std::cell::Cell;
    // TODO: profile before using a different non crypto PRNG.
    //       True entropy is more important than is intuitive for search.
    use rand::{Rng, SeedableRng};
    #[test]
    pub fn construct_rot_net() {
        let mut rng = rand::thread_rng();
        // 1. initialize nodes
        let nodes = network::initialize_nodes(3, 2);
        let out_nodes = nodes.1.iter().collect::<Vec<&Node>>();
        let in_nodes = nodes.0.iter().collect::<Vec<&Node>>();

        // 2. initialize connections
        let mut rngs = vec![];
        for i in 0..out_nodes.len() {
            println!("generating a connection param rngs..");
            rngs.push(rng.gen());
        }
        // clone the references so we dont move it out until rot_net.
        let init_connections =
            network::initialize_connections(in_nodes.clone(), out_nodes.clone(), rngs);
        let connections = init_connections.iter().collect::<Vec<&Cell<Connection>>>();

        // 3. initialize rot_net
        let rot_net = network::rot_net_initialize(in_nodes, out_nodes, connections);
        // 4. cycle the network
        let res = rot_net.cycle(vec![1 as u8, 2 as u8, 3 as u8]);
        for r in res {
            println!("got result: {}", r);
        }
    }
}

fn main() {
    println!("Hello, world!");

    let mut rng = rand::thread_rng();
    // 1. initialize nodes
    let nodes = network::initialize_nodes(3, 2);
    let out_nodes = nodes.1.iter().collect::<Vec<&Node>>();
    let in_nodes = nodes.0.iter().collect::<Vec<&Node>>();

    // 2. initialize connections
    let mut rngs = vec![];
    for i in 0..out_nodes.len() {
        for j in 0..in_nodes.len() {
            rngs.push(rng.gen());
        }
    }
    // clone the references so we dont move it out until rot_net.
    let init_connections =
        network::initialize_connections(in_nodes.clone(), out_nodes.clone(), rngs);
    let connections = init_connections.iter().collect::<Vec<&Cell<Connection>>>();

    // 3. initialize rot_net
    let rot_net = network::rot_net_initialize(in_nodes, out_nodes, connections);

    // 4. cycle the network
    let res = timeit_loops!(10000000, {
        let sol = rot_net.clone().cycle(vec![rng.gen(), rng.gen(), rng.gen()]);
        // for i in sol {
        //     println!("got {}", i);
        // }
    });
    // println!("time: {}", res);
    // for r in res {
    //     println!("got result: {}", r);
    // }
}
