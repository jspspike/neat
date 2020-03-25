use indexmap::IndexMap;
use rand::Rng;

use super::genome::Genome;

pub trait Task {
    fn new(seed: u64, inputs: u16, outputs: u16) -> Self;
    fn step(&mut self, inputs: Vec<f32>) -> Vec<f32>;
    fn score(&self) -> Option<f32>;
}

struct Edge {
    start: u16,
    end: u16,
    weight: f32,
}

struct Network<T: Task> {
    task: T,
    nodes: IndexMap<u16, f32>,
    edges: Vec<Edge>,
    inputs: u16,
    outputs: u16,
}

impl<T: Task> Network<T> {
    pub fn new(genome: Genome) -> Network<T> {
        let mut rng = rand::thread_rng();

        Network {
            task: T::new(rng.gen::<u64>(), genome.inputs, genome.outputs),
            nodes: genome.nodes.iter().map(|n| (n.innovation, 0.0)).collect(),
            edges: genome
                .connections
                .iter()
                .filter(|(_, c)| c.enabled)
                .map(|((start, end), c)| Edge {
                    start: *start,
                    end: *end,
                    weight: c.weight,
                })
                .collect(),
            inputs: genome.inputs,
            outputs: genome.outputs,
        }
    }

    fn set_inputs(&mut self, inputs: Vec<f32>) {
        assert_eq!(self.inputs, inputs.len() as u16);
        for i in 0..self.inputs as usize {
            let (_, node) = self.nodes.get_index_mut(i).unwrap();
            *node = inputs[i];
        }
    }

    fn get_outputs(&self) -> Vec<f32> {
        self.nodes
            .values()
            .skip(self.inputs as usize)
            .take(self.outputs as usize)
            .map(|v| *v)
            .collect()
    }

    pub fn run(&mut self) -> f32 {
        while !self.task.score().is_some() {
            let outputs = self.get_outputs();
            let inputs = self.task.step(outputs);
            self.set_inputs(inputs);
        }

        self.task.score().unwrap()
    }
}
