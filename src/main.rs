#[macro_use]
extern crate timeit;

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
        //TODO: graph this to csv and visualize in sheets.
        //      write to buffer than flush to disk to time
        //      binops wrt. virtual machine and stack machine
        for i in 1..255 {
            print!(
                "\n sent {} and got: {} == {}",
                i,
                format!("{:b}", cond_rot_act(i)),
                cond_rot_act(i)
            );
        }
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
                i*(i+1);
            }
        });
    }
    #[test]
    fn test_connection_propagation() {
        print!("Testing forward propagation of a connection with 8 bit shift operations");
        let connection = Connection {
            weight: 4 as u8,
            direction: true,
            in_connection: None,
            out_connection: None,
        };
        for i in 1..255 {
            let solution = connection.forward_propagate(i as u32);
            // res = connection.propagate()
            print!("\n sent {} and got {}", i, solution);
        }
    }
    #[test]
    fn test_network_forward_propagation() {
        print!("Testing forward propagation of a network");
    }
}

//CURRENTLY TESTING WITH PRINTOUT:
// activation.
fn main() {

    use net::activations::cond_rot_act;
    print!("Testing of sigmoid activation of 8 bit integer using only bitshift operations.");
    print!("target runtime is 3 clock cycles for 3 barrel shift operations.\n");
    //TODO: graph this to csv and visualize in sheets.
    //      write to buffer than flush to disk to time
    //      binops wrt. virtual machine and stack machine
    for i in 1..255 {
        print!(
            "\n sent {} and got: {} == {}",
            i,
            format!("{:b}", cond_rot_act(i)),
            cond_rot_act(i)
        );
    }
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
            i*(i+1);
        }
    });
}
