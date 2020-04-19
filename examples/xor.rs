use neat::{Neat, NeatSettings, Network, Task};

struct Xor {
    count: u8,
    score: u8,
}

impl Task for Xor {
    fn new(_: u64) -> Xor {
        Xor { count: 0, score: 0 }
    }

    fn step(&mut self, inputs: Vec<f32>) -> Vec<f32> {
        assert_eq!(inputs.len(), 1);

        let output = match self.count {
            0 => vec![0.0, 0.0],
            1 => {
                if inputs[0] as u8 == 0 {
                    self.score += 1;
                }
                vec![0.0, 1.0]
            }
            2 => {
                if inputs[0] as u8 == 1 {
                    self.score += 1;
                }
                vec![1.0, 0.0]
            }
            3 => {
                if inputs[0] as u8 == 1 {
                    self.score += 1;
                }
                vec![1.0, 1.0]
            }
            4 => {
                if inputs[0] as u8 == 0 {
                    self.score += 1;
                }
                vec![0.0, 0.0]
            }
            _ => vec![0.0, 0.0],
        };

        self.count += 1;
        output
    }

    fn score(&self) -> Option<f32> {
        if self.count > 4 {
            Some(self.score as f32)
        } else {
            None
        }
    }
}

fn main() {
    let mut settings = NeatSettings::default();
    settings.add_node_rate = 0.9;
    settings.species_threshold = 0.6;
    settings.weight_mutate = 2.0;
    let mut neat = Neat::<Xor>::new(1000, 2, 1, settings);

    let mut network = None;
    let mut fitness = 0.0;

    for _ in 0..1000 {
        let best = neat.step();
        network = Some(best.0);
        fitness = best.1;

        if fitness == 4.0 {
            break;
        }
    }

    dbg!(network);
    dbg!(fitness);
}
