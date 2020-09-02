use indexmap::IndexMap;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

use super::genome::Genome;

/// Task that can be executed by `Network` and train `Neat`
pub trait Task {
    /// Constructor to create instance of `Task`
    ///
    /// # Arguments
    ///
    /// * `seed` - Random seed to be used in Tasks that use random generators
    fn new(seed: u64) -> Self;
    /// Function to execute step of `Task` and returns outputs
    ///
    /// # Arguments
    ///
    /// * `inputs` - Inputs for `Task`
    fn step(&mut self, inputs: Vec<f32>) -> Vec<f32>;
    /// Returns score or fitness of `Task`. Should return `None` if `Task` is still ongoing and
    /// should return `Some` with score when completed
    fn score(&self) -> Option<f32>;
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Edge {
    start: u16,
    weight: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Node {
    value: f32,
    activation: f32,
    inputs: Vec<Edge>,
}

/// Neural network used to execute tasks
#[derive(Debug, Serialize, Deserialize)]
pub struct Network {
    nodes: IndexMap<u16, Node>,
    inputs: u16,
    outputs: u16,
}

fn sigmoid(x: f32, activation: f32) -> f32 {
    1.0 / (1.0 + (-activation * x).exp())
}

impl Network {
    pub(crate) fn new(genome: Genome) -> Network {
        let mut nodes: IndexMap<u16, Node> = genome
            .nodes
            .iter()
            .map(|(i, n)| {
                (
                    *i,
                    Node {
                        value: 0.0,
                        activation: n.activation,
                        inputs: Vec::new(),
                    },
                )
            })
            .collect();

        for connection in genome.connections.iter().filter(|(_, c)| c.enabled) {
            let ((start, end), conn) = connection;
            let node = nodes.get_mut(end).unwrap();
            node.inputs.push(Edge {
                start: *start,
                weight: conn.weight,
            });
        }

        Network {
            nodes,
            inputs: genome.inputs,
            outputs: genome.outputs,
        }
    }

    fn set_inputs(&mut self, inputs: Vec<f32>) {
        assert_eq!(self.inputs, inputs.len() as u16);
        for (i, value) in inputs.iter().enumerate().take(self.inputs as usize) {
            let (_, node) = self.nodes.get_index_mut(i).unwrap();
            node.value = *value;
        }
    }

    /// Retrieve outputs of `Network`, should match length of inputs for `Task`
    pub fn get_outputs(&self) -> Vec<f32> {
        self.nodes
            .values()
            .skip(self.inputs as usize)
            .take(self.outputs as usize)
            .map(|v| v.value)
            .collect()
    }

    /// Reset all node values to 0.0
    pub fn reset(&mut self) {
        for (_, node) in self.nodes.iter_mut() {
            node.value = 0.0;
        }
    }

    fn eval(&mut self, node: Node, solved: &mut HashSet<u16>) -> f32 {
        let mut val = 0.0;

        for edge in node.inputs {
            val += if solved.contains(&edge.start) {
                self.nodes[&edge.start].value * edge.weight
            } else {
                let n = self.nodes[&edge.start].clone();

                solved.insert(edge.start);
                let v = self.eval(n, solved);

                self.nodes[&edge.start].value = v;
                v * edge.weight
            }
        }

        sigmoid(val, node.activation)
    }

    /// Propagate inputs throughout network
    ///
    /// # Arguments
    ///
    /// * `inputs` - Input values for network
    pub fn prop(&mut self, inputs: Vec<f32>) {
        self.set_inputs(inputs);

        let mut solved: HashSet<u16> = HashSet::new();
        for i in 0..self.inputs {
            solved.insert(i as u16);
        }

        for i in self.inputs..(self.inputs + self.outputs) {
            let node = self.nodes[&i].clone();
            self.nodes[&i].value = self.eval(node, &mut solved);
        }
    }

    /// Run given `Task` to completion using network. This will take `Network` outputs and use them as inputs in `Task` `step`. Then run `prop` using `Task` outputs. Once `Task` `score` returns `Some`, execution will be stopped and the score from `Task` will be returned.
    pub fn run<T: Task>(&mut self) -> f32 {
        let mut rng = rand::thread_rng();
        let mut task = T::new(rng.gen::<u64>());

        while task.score().is_none() {
            let outputs = self.get_outputs();
            let inputs = task.step(outputs);
            self.prop(inputs);
        }

        task.score().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genome::*;

    struct Test {
        count: u8,
    }

    impl Task for Test {
        fn new(_: u64) -> Test {
            Test { count: 0 }
        }

        fn step(&mut self, inputs: Vec<f32>) -> Vec<f32> {
            assert_eq!(inputs.len(), 1);
            match self.count {
                0 => assert_eq!(inputs[0], 0.0),
                1 => assert_eq!(inputs[0], 0.5),
                2 => assert_eq!(inputs[0], 0.5),
                _ => {}
            }

            self.count += 1;

            vec![4.0]
        }

        fn score(&self) -> Option<f32> {
            if self.count < 3 {
                None
            } else {
                Some(3.0)
            }
        }
    }

    #[test]
    fn test_network() {
        let mut connections = IndexMap::new();
        connections.insert(
            (0, 28),
            Connection {
                weight: -3.0,
                enabled: true,
            },
        );
        connections.insert(
            (28, 1),
            Connection {
                weight: -7.0,
                enabled: true,
            },
        );

        let mut nodes = IndexMap::new();

        nodes.insert(0, Neuron { activation: 4.9 });
        nodes.insert(1, Neuron { activation: 4.9 });
        nodes.insert(28, Neuron { activation: 4.9 });

        let genome = Genome {
            inputs: 1,
            outputs: 1,
            nodes,
            connections,
        };

        let mut network: Network = Network::new(genome);
        assert_eq!(network.run::<Test>(), 3.0);
    }
}
