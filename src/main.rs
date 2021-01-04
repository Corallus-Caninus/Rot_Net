#[macro_use]
extern crate timeit;

fn main() {}

#[cfg(test)]
mod tests {
    use net::activations::*;
    use log::info;

    #[test]
    fn test_activation() {
        info!("This is a test of sigmoid activation of 8 bit integer using exclusive bitshift operations.");
        info!("target runtime is 3 clock cycles for 3 barrel shift operations.\n");
        //TODO: graph this to csv and visualize in sheets.
        //      write to buffer than flush to disk to time
        //      binops wrt. virtual machine and stack machine
        for i in 1..255 {
            info!(
                "\n sent {} and got: {} == {} \n",
                i,
                format!("{:b}", cond_rot_act(i)),
                cond_rot_act(i)
            );
        }
        info!("testing conditional rotation activation..");
        timeit!({
            for i in 1..255 {
                cond_rot_act(i);
            }
        });
        info!("testing unoptimized full 32x precision sigmoid activation..");
        timeit!({
            for i in 1..255 {
                (1.0_f32 / 1.0_f32 + (-1.0_f32 * (i as f32)).exp());
            }
        })
    }
}
