#[macro_use]
extern crate timeit;

fn main() {}

#[cfg(test)]
mod tests {
    use net::activations::*;
    use net::connections::*;
    use log::info;

    #[test]
    fn test_activation() {
        info!("Testing of sigmoid activation of 8 bit integer using only bitshift operations.");
        info!("target runtime is 3 clock cycles for 3 barrel shift operations.\n");
        //TODO: graph this to csv and visualize in sheets.
        //      write to buffer than flush to disk to time
        //      binops wrt. virtual machine and stack machine
        for i in 1..255 {
            info!(
                "\n sent {} and got: {} == {}",
                i,
                format!("{:b}", cond_rot_act(i)),
                cond_rot_act(i)
            );
        }
        info!("Testing conditional rotation activation..");
        timeit!({
            for i in 1..255 {
                cond_rot_act(i);
            }
        });
        info!("Testing unoptimized full 32x precision sigmoid activation..");
        timeit!({
            for i in 1..255 {
                (1.0_f32 / 1.0_f32 + (-1.0_f32 * (i as f32)).exp());
            }
        })
    }
    #[test]
    fn test_connection_propagation(){
        info!("Testing forward propagation of a connection with 8 bit shift operations");
        let connection = Connection{weight: 4 as u8,direction: true,
            in_connection: None,out_connection: None};
        for i in 1..255{
            let solution = connection.propagate(i as u8);
            // res = connection.propagate()
            info!("\n sent {} and got {}", i, solution);
        }
    }
    #[test]
    fn test_network_forward_propagation{
        info!("Testing forward propagation of a network")
    }
}
