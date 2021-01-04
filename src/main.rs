use net::activations::cond_rot_act;

#[macro_use]
extern crate timeit;
/// This will be a network where weights are sig^weight instead of sig*weight
/// activations will be the rot_sig_asym (asymmetric sigmoid using bitshifts)

//TODO: test driven development this with unittests as a practice
//      (project isnt big enough to justify but need to learn)
// TODO: this is all deprecated
fn main() {}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_activation() {
        println!("This is a test of sigmoid activation of 8 bit integer using exclusive bitshift operations.");
        println!("target runtime is 3 clock cycles for 3 barrel shift operations.\n");
        //TODO: graph this to csv and visualize in sheets.
        //      write to buffer than flush to disk to time
        //      binops wrt. virtual machine and stack machine
        //for i in 0..4294967295{
        for i in 1..255 {
            //if i%100000==0{
            //println!("\nsent {} got: {}\n",i,rot_act(i));
            println!(
                "\n sent {} and got: {} == {} \n",
                i,
                format!("{:b}", cond_rot_act(i)),
                cond_rot_act(i)
            );
            //println!("\ngot: {}\n",format!("{:b}",rot_act(i)));
            //}
        }
        println!("testing conditional rotation activation..");
        timeit!({
            for i in 1..255 {
                cond_rot_act(i);
            }
        });
        println!("testing unoptimized full 32x precision sigmoid activation..");
        timeit!({
            for i in 1..255 {
                (1.0_f32 / 1.0_f32 + (-1.0_f32 * (i as f32)).exp());
            }
        })
        //println!("last got: {}",rot_act(4294967295/2));
    }
}
