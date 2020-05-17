use neat::{Neat, Task};

struct Xor {
    count: u8,
    score: f32,
}

impl Task for Xor {
    fn new(_: u64) -> Xor {
        Xor {
            count: 0,
            score: 0.0,
        }
    }

    fn step(&mut self, inputs: Vec<f32>) -> Vec<f32> {
        assert_eq!(inputs.len(), 1);

        let output = match self.count {
            0 => vec![0.0, 0.0],
            1 => {
                if inputs[0] < 0.1 {
                    self.score += 1.0;
                } else if inputs[0] < 1.0 {
                    self.score += 1.0 - inputs[0];
                }
                vec![0.0, 1.0]
            }
            2 => {
                if inputs[0] > 0.9 {
                    self.score += 1.0;
                } else if inputs[0] > 0.0 {
                    self.score += inputs[0];
                }
                vec![1.0, 0.0]
            }
            3 => {
                if inputs[0] > 0.9 {
                    self.score += 1.0;
                } else if inputs[0] > 0.0 {
                    self.score += inputs[0];
                }
                vec![1.0, 1.0]
            }
            4 => {
                if inputs[0] < 0.1 {
                    self.score += 1.0;
                } else if inputs[0] < 1.0 {
                    self.score += 1.0 - inputs[0];
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
            Some(self.score)
        } else {
            None
        }
    }
}

fn main() {
    let mut neat = Neat::<Xor>::default(1000, 2, 1);

    let mut best = neat.step();

    for i in 0..500 {
        best = neat.step();
        if best.1 >= 4.0 {
            dbg!(i);
            break;
        }
    }

    let (mut network, fitness) = best;

    dbg!(fitness);
    dbg!(&network);

    network.reset();
    network.prop(vec![0.0, 0.0]);
    dbg!(network.get_outputs());
    network.prop(vec![0.0, 1.0]);
    dbg!(network.get_outputs());
    network.prop(vec![1.0, 0.0]);
    dbg!(network.get_outputs());
    network.prop(vec![1.0, 1.0]);
    dbg!(network.get_outputs());
}
