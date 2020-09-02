//! # Quick Start
//!
//! Implement `Task` on the struct containing the logic for the task you want to train using NEAT.
//! An example of this can be found here [examples/snake.rs](https://github.com/jspspike/neat/blob/master/examples/snake.rs).
//! Then use `Neat` to train on this task.
//! ```ignore
//! use neat::Neat;
//!
//! let mut neat = Neat::<ImplementedTask>::default(1000, 2, 1);
//!
//! // `step` will execute and train on one generation of genomes.
//! // It returns the network and fitness of the most fit genome in that step
//! let (network, fitness) = neat.step();
//! ```
//!
//! Finally you can use the `Network` to execute your task. If you have the struct that implements
//! `Task` you can pass that to it directly or use the `prop` function to get one step. You can
//! find an example of this in [examples/run-snake.rs](https://github.com/jspspike/neat/blob/master/examples/run-snake.rs)
//! ```ignore
//! use neat::Network;
//!
//! let mut inputs = !vec[0.0];
//!
//! loop {
//!     let outputs = network.prop(inputs);
//!     inputs = task.do_stuff(outputs);
//! }
//! ```

mod genome;
mod innovation;
mod neat;
mod network;

pub use crate::neat::Neat;
pub use crate::neat::NeatSettings;
pub use network::Network;
pub use network::Task;
