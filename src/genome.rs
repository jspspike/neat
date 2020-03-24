use indexmap::IndexMap;
use rand::Rng;

use super::innovation::InnovationCounter;
use super::neat::NeatSettings;

struct Node {
    innovation: u16,
}

struct Connection {
    weight: f32,
    enabled: bool,
}

pub(super) struct Genome {
    inputs: u16,
    outputs: u16,
    nodes: Vec<Node>,
    connections: IndexMap<(u16, u16), Connection>,
}

impl Genome {
    pub fn new(inputs: u16, outputs: u16) -> Genome {
        let mut nodes: Vec<Node> = vec![];
        for i in 0..inputs {
            nodes.push(Node { innovation: i });
        }
        for i in inputs..inputs + outputs {
            nodes.push(Node { innovation: i })
        }

        Genome {
            inputs,
            outputs,
            nodes,
            connections: IndexMap::new(),
        }
    }

    fn is_output(&self, index: usize) -> bool {
        index >= self.inputs as usize && index < self.outputs as usize
    }

    fn add_connection(
        &mut self,
        innovations: &mut InnovationCounter,
        settings: &NeatSettings,
    ) -> bool {
        let mut rng = rand::thread_rng();

        let input = rng.gen_range(0, self.nodes.len());
        let output = rng.gen_range(self.inputs as usize, self.nodes.len());

        let connection = (self.nodes[input].innovation, self.nodes[output].innovation);

        if (self.is_output(input) && self.is_output(output))
            || !self.connections.get(&connection).is_some()
        {
            return false;
        }

        self.connections.insert(
            connection,
            Connection {
                weight: rng.gen_range(-settings.weight, settings.weight),
                enabled: true,
            },
        );
        innovations.add(connection);

        return true;
    }

    fn add_node(&mut self, innovations: &InnovationCounter) {
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

        self.connections.insert(
            (start, innovation),
            Connection {
                weight: 1.0,
                enabled: true,
            },
        );
        self.connections.insert(
            (innovation, end),
            Connection {
                weight: weight,
                enabled: true,
            },
        );

        self.nodes.push(Node { innovation });
    }

    fn mutate_connections(&mut self, settings: &NeatSettings) {
        let mut rng = rand::thread_rng();

        for (_, info) in self.connections.iter_mut().filter(|(_, i)| i.enabled) {
            if rng.gen::<f32>() <= settings.weight_mutate_rate {
                info.weight += rng.gen_range(-settings.weight_mutate, settings.weight_mutate);
            }
        }
    }

    pub fn mutate(&mut self, mut innovations: InnovationCounter, settings: NeatSettings) {
        let mut rng = rand::thread_rng();

        if rng.gen::<f32>() <= settings.add_connection_rate {
            self.add_connection(&mut innovations, &settings);
        }

        if rng.gen::<f32>() <= settings.add_node_rate {
            self.add_node(&innovations);
        }

        self.mutate_connections(&settings);
    }
}
