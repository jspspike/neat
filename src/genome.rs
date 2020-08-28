use indexmap::IndexMap;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::cmp::max;

use super::innovation::InnovationCounter;
use super::neat::NeatSettings;

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub(crate) struct Neuron {
    pub(crate) activation: f32,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub(crate) struct Connection {
    pub(crate) weight: f32,
    pub(crate) enabled: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct Genome {
    pub(crate) inputs: u16,
    pub(crate) outputs: u16,
    pub(crate) nodes: IndexMap<u16, Neuron>,
    pub(crate) connections: IndexMap<(u16, u16), Connection>,
}

impl Genome {
    pub(crate) fn new(inputs: u16, outputs: u16) -> Genome {
        let mut nodes = IndexMap::new();
        for i in 0..inputs {
            nodes.insert(i, Neuron { activation: 4.9 });
        }
        for i in inputs..inputs + outputs {
            nodes.insert(i, Neuron { activation: 4.9 });
        }

        Genome {
            inputs,
            outputs,
            nodes,
            connections: IndexMap::new(),
        }
    }

    fn is_output(&self, index: usize) -> bool {
        index >= self.inputs as usize && index < (self.inputs + self.outputs) as usize
    }

    pub(crate) fn add_connection(
        &mut self,
        innovations: &mut InnovationCounter,
        settings: &NeatSettings,
    ) -> bool {
        let mut rng = rand::thread_rng();

        let input = rng.gen_range(0, self.nodes.len());
        let output = rng.gen_range(self.inputs as usize, self.nodes.len());

        if input == output || (self.is_output(input) && self.is_output(output)) {
            return false;
        }

        let (input_node, _) = self.nodes.get_index(input).unwrap();
        let (output_node, _) = self.nodes.get_index(output).unwrap();

        let connection = (*input_node, *output_node);
        let reverse = (*output_node, *input_node);

        if settings.feedforward && self.connections.contains_key(&reverse) {
            return false;
        }

        if let Some(info) = self.connections.get_mut(&connection) {
            info.enabled = true;
            return true;
        }

        self.connections.insert(
            connection,
            Connection {
                weight: rng.gen_range(-settings.weight, settings.weight),
                enabled: true,
            },
        );
        innovations.add(connection);

        true
    }

    fn add_node(&mut self, innovations: &mut InnovationCounter) {
        if self.connections.is_empty() {
            return;
        }

        let mut rng = rand::thread_rng();

        let (connection, info) = self
            .connections
            .get_index_mut(rng.gen_range(0, self.connections.len()))
            .expect("Should not request invalid connection");

        let innovation = innovations
            .get(*connection)
            .expect("Should not request invalid innovation");

        info.enabled = false;
        let (start, end) = *connection;
        let weight = info.weight;

        innovations.add((start, innovation));
        self.connections.insert(
            (start, innovation),
            Connection {
                weight: 1.0,
                enabled: true,
            },
        );

        innovations.add((innovation, end));
        self.connections.insert(
            (innovation, end),
            Connection {
                weight,
                enabled: true,
            },
        );

        self.nodes.insert(innovation, Neuron { activation: 4.9 });
    }

    fn mutate_connections(&mut self, settings: &NeatSettings) {
        let mut rng = rand::thread_rng();

        for (_, info) in self.connections.iter_mut().filter(|(_, i)| i.enabled) {
            if rng.gen::<f32>() < settings.weight_mutate_rate {
                info.weight += rng.gen_range(-settings.weight_mutate, settings.weight_mutate);
            }
        }
    }

    fn mutate_nodes(&mut self, settings: &NeatSettings) {
        let mut rng = rand::thread_rng();

        for (_, node) in self.nodes.iter_mut() {
            if rng.gen::<f32>() <= settings.activation_mutate_rate {
                node.activation +=
                    rng.gen_range(-settings.activation_mutate, settings.activation_mutate);
            }
        }
    }

    pub(crate) fn mutate(&mut self, innovations: &mut InnovationCounter, settings: &NeatSettings) {
        let mut rng = rand::thread_rng();

        if rng.gen::<f32>() <= settings.add_connection_rate {
            self.add_connection(innovations, settings);
        }

        if rng.gen::<f32>() <= settings.add_node_rate {
            self.add_node(innovations);
        }

        self.mutate_connections(&settings);

        self.mutate_nodes(&settings);
    }

    pub(crate) fn cross(better: &Genome, worse: &Genome) -> Genome {
        assert_eq!(better.inputs, worse.inputs);
        assert_eq!(better.outputs, worse.outputs);

        let mut child = Genome::new(better.inputs, better.outputs);

        for (connection, info) in better.connections.iter() {
            let (start, end) = connection;

            child.connections.insert(
                *connection,
                if worse.connections.contains_key(connection)
                    && rand::thread_rng().gen::<f32>() < 0.5
                {
                    if !child.nodes.contains_key(start) {
                        child.nodes.insert(*start, *worse.nodes.get(start).unwrap());
                    }

                    if !child.nodes.contains_key(end) {
                        child.nodes.insert(*end, *worse.nodes.get(end).unwrap());
                    }

                    *worse.connections.get(connection).unwrap()
                } else {
                    if !child.nodes.contains_key(start) {
                        child
                            .nodes
                            .insert(*start, *better.nodes.get(start).unwrap());
                    }

                    if !child.nodes.contains_key(end) {
                        child.nodes.insert(*end, *better.nodes.get(end).unwrap());
                    }

                    *info
                },
            );
        }

        child
    }

    pub(crate) fn same_species(first: &Genome, second: &Genome, settings: &NeatSettings) -> bool {
        let mut c_diff = 0.0;
        let mut w_diff = 0.0;

        for (connection, f_info) in first.connections.iter() {
            if let Some(s_info) = second.connections.get(connection) {
                w_diff += (f_info.weight - s_info.weight).abs();
            } else {
                c_diff += 1.0;
            }
        }

        c_diff += second.connections.keys().fold(0.0, |acc, conn| {
            if first.connections.get(conn).is_none() {
                acc + 1.0
            } else {
                acc
            }
        });

        let size = max(first.connections.len(), second.connections.len());
        let connection_diff = if size != 0 {
            (c_diff * settings.connections_diff) / size as f32
        } else {
            0.0
        };

        let weight_diff = w_diff * settings.weight_diff;

        (connection_diff + weight_diff) < settings.species_threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_same_species() {
        let settings = NeatSettings {
            weight: 1.0,
            weight_mutate: 1.0,
            weight_max: 10.0,
            weight_mutate_rate: 0.8,
            add_connection_rate: 0.2,
            add_node_rate: 0.1,
            activation_mutate: 0.05,
            activation_mutate_rate: 0.1,
            connections_diff: 1.0,
            weight_diff: 0.1,
            species_threshold: 1.0,
            feedforward: true,
        };

        let mut first = Genome::new(1, 2);
        first.connections.insert(
            (0, 2),
            Connection {
                weight: 0.5,
                enabled: true,
            },
        );
        let mut second = Genome::new(1, 2);
        second.connections.insert(
            (0, 2),
            Connection {
                weight: 1.5,
                enabled: true,
            },
        );

        assert!(Genome::same_species(&first, &second, &settings));

        second.connections.insert(
            (0, 1),
            Connection {
                weight: 1.5,
                enabled: true,
            },
        );
        second.connections.insert(
            (0, 22),
            Connection {
                weight: 1.5,
                enabled: true,
            },
        );
        second.connections.insert(
            (0, 21),
            Connection {
                weight: 1.5,
                enabled: true,
            },
        );
        first.connections.insert(
            (3, 22),
            Connection {
                weight: 1.5,
                enabled: true,
            },
        );

        assert!(!Genome::same_species(&first, &second, &settings));
    }
}
