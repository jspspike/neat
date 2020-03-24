pub struct NeatSettings {
    pub weight: f32,
    pub weight_mutate: f32,
    pub weight_max: f32,
    pub weight_mutate_rate: f32,
    pub add_connection_rate: f32,
    pub add_node_rate: f32,
}

impl NeatSettings {
    pub fn default() -> NeatSettings {
        NeatSettings {
            weight: 2.0,
            weight_mutate: 1.0,
            weight_max: 10.0,
            weight_mutate_rate: 0.8,
            add_connection_rate: 0.2,
            add_node_rate: 0.1,
        }
    }
}
